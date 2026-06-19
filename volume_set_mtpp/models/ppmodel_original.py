import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

MAX_SAMPLE_BATCH_SIZE = 1024

class PPModel(nn.Module):
    def __init__(
            self,
            decoder,
            num_channels,
            channel_embedding,
            dominating_rate=10000.,
            dyn_dom_buffer=4,
    ):
        """Constructor for general PPModel class.

        Arguments:
            decoder {torch.nn.Module} -- Neural network decoder that accepts a latent state, marks, timestamps, and times of sample points.
        """
        super().__init__()

        self.decoder = decoder
        self.num_channels = num_channels
        self.channel_embedding = channel_embedding
        self.dominating_rate = dominating_rate
        self.dyn_dom_buffer = dyn_dom_buffer

    def get_states(self, marks, timestamps, old_states=None):
        """Get the hidden states that can be used to extract intensity values from."""

        states = self.decoder.get_states(
            marks=marks,
            timestamps=timestamps,
            old_states=old_states,
        )
        return {
            "state_values": states,
            "state_times": timestamps,
        }

    def get_intensity(self, state_values, state_times, timestamps, marks=None, state_marks=None):
        """Given a set of hidden states, timestamps, and latent_state get a tensor representing intensity values at timestamps.
        Get the total intensity for a point process and item specific parameters."""

        if (state_values is None) and (state_marks is not None):
            state_values = self.get_states(state_marks, state_times)["state_values"]

        h_t = self.decoder.get_hidden_h(
            state_values=state_values,
            state_times=state_times,
            timestamps=timestamps
        )
        intensity_dict = self.get_total_intensity_and_items(h_t)
        return intensity_dict

    @abstractmethod
    def get_total_intensity_and_items(self, h_t):
    # @abstractmethod
    # def get_total_intensity_and_items(self, h_t, time_logits=None):
        raise NotImplementedError("Must override get_total_intensity_and_items")

    def forward(self, marks, timestamps, sample_timestamps=None):
        """Encodes a(n optional) set of marks and timestamps into a latent vector,
        then decodes corresponding intensity values for a target set of timestamps and marks
        (as well as a sample set if specified).

        Arguments:
            ref_marks {torch.LongTensor} -- Tensor containing mark ids that correspond to channel embeddings. Part of the reference set to be encoded.
            ref_timestamps {torch.FloatTensor} -- Tensor containing times that correspond to the events in `ref_marks`. Part of the reference set to be encoded.
            ref_marks_bwd {torch.LongTensor} -- Tensor containing reverse mark ids that correspond to channel embeddings. Part of the reference set to be encoded.
            ref_timestamps_bwd {torch.FloatTensor} -- Tensor containing reverse times that correspond to the events in `ref_marks`. Part of the reference set to be encoded.
            tgt_marks {torch.FloatTensor} -- Tensor containing mark ids that correspond to channel embeddings. These events will be decoded and are assumed to have happened.
            tgt_timestamps {torch.FloatTensor} -- Tensor containing times that correspond to the events in `tgt_marks`. These times will be decoded and are assumed to have happened.
            context_lengths {torch.LongTensor} -- Tensor containing position ids that correspond to last events in the reference material.

        Keyword Arguments:
            sample_timestamps {torch.FloatTensor} -- Times that will have intensity values generated for. These events are _not_ assumed to have happened. (default: {None})

        Returns:
            dict -- Dictionary containing the produced latent vector, intermediate hidden states, and intensity values for target sequence and sample points.
        """
        return_dict = {}
        if marks is None:
            marks = torch.LongTensor([[]], device=next(self.parameters()).device)
            timestamps = torch.FloatTensor([[]], device=next(self.parameters()).device)

        # Decoding phase
        intensity_state_dict = self.get_states(
            marks=marks,
            timestamps=timestamps,
        )
        return_dict["state_dict"] = intensity_state_dict

        intensities = self.get_intensity(
            state_values=intensity_state_dict["state_values"],
            state_times=intensity_state_dict["state_times"],
            timestamps=timestamps,
        )
        return_dict["intensities"] = intensities

        # Sample intensities for objective function
        if sample_timestamps is not None:
            sample_intensities = self.get_intensity(
                state_values=intensity_state_dict["state_values"],
                state_times=intensity_state_dict["state_times"],
                timestamps=sample_timestamps,
                marks=None,
            )
            return_dict["sample_intensities"] = sample_intensities

        return return_dict


    def sample_points(self, marks, timestamps, dominating_rate=None, T=float('inf'), left_window=0.0,
                      length_limit=float('inf'), mark_mask=None, proposal_batch_size=1024):
        assert ((T < float('inf')) or (length_limit < float('inf')))
        if dominating_rate is None:
            dominating_rate = self.dominating_rate
        if marks is None:
            # marks = torch.LongTensor([[]], device=next(self.parameters()).device)
            marks = torch.FloatTensor([[]], device=next(self.parameters()).device)
            timestamps = torch.FloatTensor([[]], device=next(self.parameters()).device)

        if T < float('inf'):
            proposal_batch_size = max(min(proposal_batch_size, int(dominating_rate * (T - left_window) * 5)),
                                      10)  # dominating_rate*(T-left_window) is the expected number of proposal times to draw from [left_window, T]

        state = self.forward(marks, timestamps)
        state_values, state_times = state["state_dict"]["state_values"], state["state_dict"]["state_times"]

        dist = torch.distributions.Exponential(dominating_rate)
        dist.rate = dist.rate.to(state_values.device)
        last_time = left_window

        new_times = last_time + dist.sample(sample_shape=torch.Size((1, proposal_batch_size))).cumsum(dim=-1)
        sampled_times = []
        sampled_marks = []

        while (new_times <= T).any() and (timestamps.shape[-1] < length_limit):
            new_times = new_times[new_times <= T].unsqueeze(0)
            sample_intensities = self.get_intensity(
                state_values=state_values,
                state_times=state_times,
                timestamps=new_times,
                marks = None,
            )

            acceptances = torch.rand_like(new_times) <= (sample_intensities["total_intensity"].squeeze(-1) / dominating_rate)
            if acceptances.any():
                idx = acceptances.squeeze(0).float().argmax()  # first occurrence of 1 in boolean 'acceptances'
                new_time = new_times[:, [idx]]
                new_mark = self.sample_single_set(sample_intensities, idx)
                timestamps = torch.cat((timestamps, new_time), -1)

                # should be torch.Size([batch_size, num_events, num_channel])
                if marks.numel() == 0:
                    marks = torch.cat((marks.unsqueeze(0), new_mark), -1)
                else:
                    # append events for a single sequence
                    marks = torch.cat((marks, new_mark), -2)  # tensor([[new_mark_multi_hot_t0],[new_mark_multi_hot_t1],...])  # Size([batch, event, channel])

                sampled_times.append(new_time.squeeze().item())
                sampled_marks.append(new_mark)
                # sampled_marks.append(new_mark.squeeze().item())

                state = self.forward(marks, timestamps)
                state_values, state_times = state["state_dict"]["state_values"], state["state_dict"]["state_times"]
                last_time = new_times[:, idx].squeeze()
            else:
                last_time = new_times.max()

            new_times = last_time + dist.sample(sample_shape=(1, proposal_batch_size)).cumsum(dim=-1)

        assumption_violation = False
        for _ in range(5):
            eval_times = torch.rand_like(timestamps).clamp(min=1e-8) * T
            sample_intensities = self.get_intensity(
                state_values=state_values,
                state_times=state_times,
                timestamps=eval_times,
                marks=None,
            )
            if (sample_intensities["total_intensity"] > dominating_rate).any().item():
                print("DR: {}".format(dominating_rate))
                print("IN: {}".format(sample_intensities["total_intensity"].max().item()))
                assumption_violation = True
                break
        if assumption_violation:
            print("Violation in sampling assumption occurred. Redoing sample.")
            return None  # self.sample_points(ref_marks, ref_timestamps, ref_marks_bwd, ref_timestamps_bwd, tgt_marks, tgt_timestamps, context_lengths, dominating_rate * 2, T)
        else:
            return (timestamps, marks)  # timestamps: torch.Size([1, num_time]), marks in multi-hot, but include the whole seq
            # return (sampled_times, sampled_marks)

    def determine_mark_mask(self, new_times, sample_lens, mask_dict):
        if "temporal_mark_restrictions" in mask_dict:
            mark_masks = mask_dict["temporal_mark_restrictions"]  # (num_boundaries+1, num_channels)
            time_boundaries = mask_dict["time_boundaries"]  # (num_boundaries,)
            idx = (new_times.unsqueeze(-1) >= time_boundaries.unsqueeze(-2)).sum(dim=-1)  # decide which time span each of new_times falls into
        else:
            raise NotImplementedError
        return F.embedding(idx, mark_masks)  # (batch_sample_size, num_proposed_new_time, num_channel)


    def batch_sample_points(self, marks, timestamps, dominating_rate=None, T=float('inf'), left_window=0.0,
                            length_limit=float('inf'), mark_mask=None, proposal_batch_size=1024, num_samples=1,
                            mask_dict=None, adapt_dom_rate=True, stop_marks=None, censoring=None):
        dyn_dom_buffer = self.dyn_dom_buffer
        if num_samples > MAX_SAMPLE_BATCH_SIZE:  # Split into batches
            resulting_times, resulting_marks, resulting_states = [], [], []
            remaining_samples = num_samples
            while remaining_samples > 0:
                current_batch_size = min(remaining_samples, MAX_SAMPLE_BATCH_SIZE)
                sampled_times, sampled_marks, sampled_states = self.batch_sample_points(
                    marks=marks,
                    timestamps=timestamps,
                    dominating_rate=dominating_rate,
                    T=T,
                    left_window=left_window,
                    length_limit=length_limit,
                    mark_mask=mark_mask,
                    proposal_batch_size=proposal_batch_size,
                    num_samples=current_batch_size,
                    mask_dict=mask_dict,
                    adapt_dom_rate=adapt_dom_rate,
                    stop_marks=stop_marks,
                    censoring=censoring,
                )
                remaining_samples -= current_batch_size
                resulting_times.extend(sampled_times)
                resulting_marks.extend(sampled_marks)
                resulting_states.extend(sampled_states)
            return resulting_times, resulting_marks, resulting_states

        stop_for_marks = stop_marks is not None
        assert ((T < float('inf')) or (length_limit < float('inf')) or stop_for_marks)
        if mask_dict is None:
            mask_dict = {}
        if dominating_rate is None:
            dominating_rate = self.dominating_rate
        if marks is None:
            marks = torch.transpose(torch.tensor([[[]]], dtype=torch.float, device=next(self.parameters()).device), dim0=-1, dim1=-2).expand(-1, -1, self.num_channels)
            timestamps = torch.tensor([[]], dtype=torch.float, device=next(self.parameters()).device)
        if isinstance(left_window, torch.Tensor):
            left_window = left_window.item()
        if isinstance(T, torch.Tensor):
            T = T.item()
        if length_limit == float('inf'):
            length_limit = torch.iinfo(torch.int64).max  # Maximum Long value


        sample_lens = torch.zeros((num_samples,), dtype=torch.int64).to(next(self.parameters()).device) + timestamps.numel()
        marks, timestamps = marks.expand(num_samples, *marks.shape[1:]), timestamps.expand(num_samples, *timestamps.shape[1:])  # dim[0] is for batch
        time_pad, mark_pad = torch.nan_to_num(torch.tensor(float('inf'), dtype=timestamps.dtype)).item(), 0
        state = self.forward(marks, timestamps)
        state_values, state_times = state["state_dict"]["state_values"], state["state_dict"]["state_times"]
        batch_idx = torch.arange(num_samples).to(state_values.device)
        finer_proposal_batch_size = max(proposal_batch_size // 4, 16)


        dist = torch.distributions.Exponential(dominating_rate)
        dist.rate = dist.rate.to(state_values.device) * 0 + 1  # We will manually apply the scale to samples  # TODO: check this?
        dominating_rate = torch.ones((num_samples, 1), dtype=torch.float32).to(state_values.device) * dominating_rate
        last_time = torch.ones_like(dominating_rate) * left_window if isinstance(left_window, (int, float)) else left_window
        new_times = last_time + dist.sample(sample_shape=torch.Size((num_samples, proposal_batch_size))).cumsum(dim=-1) / dominating_rate
        stop_marks = stop_marks if stop_for_marks else torch.tensor([], dtype=torch.long).to(state_values.device)
        # stop_marks = torch.eye(model.num_channels)[stop_marks].sum(dim=0).to(self.device) if stop_for_marks else torch.tensor([], dtype=torch.long).to(state_values.device)
        sample_hasnt_hit_stop_marks = torch.ones((num_samples,), dtype=torch.bool).to(state_values.device)

        calculate_mark_mask = isinstance(mark_mask, type(None)) and (("temporal_mark_restrictions" in mask_dict) or ("positional_mark_restrictions" in mask_dict))
        resulting_times, resulting_marks, resulting_states = [], [], []

        if adapt_dom_rate:
            dynamic_dom_rates = torch.ones((num_samples, dyn_dom_buffer,)).to(state_values.device)*dominating_rate
            k = 0
        j = -1

        while (new_times <= T).any() and (sample_lens < length_limit).any():
            j += 1
            within_range_mask = (new_times <= T) & (sample_lens < length_limit).unsqueeze(-1) & sample_hasnt_hit_stop_marks.unsqueeze(-1)
            to_stay = within_range_mask.any(dim=-1)
            to_go = ~to_stay

            if to_go.any():
                if stop_for_marks:
                    leaving_times, leaving_marks, leaving_states = timestamps[to_go, sample_lens[to_go] - 1], marks[to_go, sample_lens[to_go] - 1, ...], state_values[to_go, sample_lens[to_go] - 1, ...]
                else:
                    leaving_times, leaving_marks, leaving_states = timestamps[to_go, ...], marks[to_go, ...], state_values[to_go, ...]
                resulting_times.append(leaving_times)
                resulting_marks.append(leaving_marks)
                resulting_states.append(leaving_states)

                new_times = new_times[to_stay, ...]
                sample_lens = sample_lens[to_stay]
                timestamps = timestamps[to_stay, ...]
                marks = marks[to_stay, ...]
                state_values = state_values[to_stay, ...]
                state_times = state_times[to_stay, ...]
                batch_idx = batch_idx[:timestamps.shape[0]]
                last_time = last_time[to_stay, ...]
                dominating_rate = dominating_rate[to_stay, ...]
                sample_hasnt_hit_stop_marks = sample_hasnt_hit_stop_marks[to_stay]
                if adapt_dom_rate:
                    dynamic_dom_rates = dynamic_dom_rates[to_stay, ...]
                if batch_idx.numel() == 0:
                    break  # STOP SAMPLING

            if calculate_mark_mask:
                mark_mask = self.determine_mark_mask(new_times, sample_lens, mask_dict).bool()
            within_range_mask = (new_times <= T) & (sample_lens < length_limit).unsqueeze(-1)

            sample_intensities = self.get_intensity(
                state_values=state_values,
                state_times=state_times,
                timestamps=new_times,
                marks=None,
            )
            # TODO: correct for CI but need to extend for other settings
            if mark_mask != None:
                weights = torch.prod(torch.where(~mark_mask, 1-sample_intensities['item_probability'], 1), dim=-1)  # only considering restricted marks
                sample_intensities["weighted_total_intensity"] = sample_intensities["total_intensity"].squeeze(-1) * weights


            redo_samples = torch.zeros_like(batch_idx, dtype=torch.bool)
            if adapt_dom_rate:  # Need to check and make sure that we don't break the sampling assumption
                if mark_mask != None:
                    redo_samples = (sample_intensities["weighted_total_intensity"] > dominating_rate).any(dim=-1)
                else:
                    redo_samples = (sample_intensities["total_intensity"].squeeze(-1) > dominating_rate).any(dim=-1)

                if not redo_samples.any():  # another finer check if total intensity violates the assumption
                    finer_new_times = last_time + dist.sample(
                        sample_shape=torch.Size((last_time.shape[0], finer_proposal_batch_size))).cumsum(dim=-1) / (
                                                  dominating_rate * proposal_batch_size / 4)
                    # print("\tFine:", dist.rate, finer_new_times.min(), finer_new_times.max(), sample_lens.min(), finer_new_times.shape[0], (finer_new_times > 3.4028e+36).any())
                    finer_mark_mask = self.determine_mark_mask(finer_new_times, sample_lens,
                                                               mask_dict) if calculate_mark_mask else None
                    finer_sample_intensities = self.get_intensity(  # Finer resolution check just after `last_time`
                        state_values=state_values,
                        state_times=state_times,
                        timestamps=finer_new_times,
                        marks=None,
                    )
                    if finer_mark_mask != None:
                        weights = torch.prod(torch.where((1 - finer_mark_mask).bool(), 1 - finer_sample_intensities['item_probability'], 1), dim=-1)  # only considering restricted marks
                        finer_sample_intensities["weighted_total_intensity"] = finer_sample_intensities["total_intensity"].squeeze(-1) * weights
                        redo_samples = redo_samples | (finer_sample_intensities["weighted_total_intensity"] > dominating_rate).any(dim=-1)
                    else:
                        redo_samples = redo_samples | (finer_sample_intensities["total_intensity"].squeeze(-1) > dominating_rate).any(dim=-1)
            keep_samples = ~redo_samples

            if mark_mask != None:
                acceptances = torch.rand_like(new_times) <= (sample_intensities["weighted_total_intensity"] / dominating_rate)
            else:
                acceptances = torch.rand_like(new_times) <= (sample_intensities["total_intensity"].squeeze(-1) / dominating_rate)  # torch.Size([num_samples, proposal_batch_size])
            acceptances = acceptances & within_range_mask & keep_samples.unsqueeze(-1)  # Don't accept any sampled events outside the window or that need to be redone
            samples_w_new_events = acceptances.any(dim=-1)


            if samples_w_new_events.any():
                event_idx = acceptances.int().argmax(dim=-1)
                new_time = new_times[batch_idx, event_idx].unsqueeze(-1)  # torch.Size([num_samples, 1])
                new_mark = self.sample_multiple_set(sample_intensities, batch_idx, event_idx, mark_mask)  # torch.Size([num_samples, num_channel])

                # Need to store sampled events into timestamps and marks
                # Some need to be appended, some need to overwrite previously written padded values
                to_append = (samples_w_new_events & (sample_lens == timestamps.shape[-1])).unsqueeze(-1)
                to_pad = ~to_append
                if to_append.any():
                    timestamps = torch.cat((timestamps, torch.where(to_append, new_time, time_pad)), -1)
                    marks = torch.cat((marks, torch.where(to_append, new_mark, mark_pad).unsqueeze(-2)), -2)

                to_overwrite = samples_w_new_events & (sample_lens < timestamps.shape[-1])
                if to_overwrite.any():
                    timestamps[to_overwrite, sample_lens[to_overwrite]] = new_time.squeeze(-1)[to_overwrite]
                    marks[to_overwrite, sample_lens[to_overwrite]] = new_mark[to_overwrite]

                sample_lens[samples_w_new_events] += 1  # Guaranteed at least one event was either appended or overwritten

                if stop_for_marks:
                    sample_hasnt_hit_stop_marks = torch.where(
                        samples_w_new_events,
                        ~(new_mark * (stop_marks == 1)).sum(dim=-1).bool(),
                        sample_hasnt_hit_stop_marks
                    )

                state = self.forward(marks, timestamps)
                state_values, state_times = state["state_dict"]["state_values"], state["state_dict"]["state_times"]
                last_time = torch.where(
                    redo_samples.unsqueeze(-1),
                    last_time,
                    torch.where(samples_w_new_events.unsqueeze(-1), new_time,
                                torch.max(new_times, dim=-1, keepdim=True).values),
                )
            else:
                last_time = torch.where(
                    redo_samples.unsqueeze(-1),
                    last_time,
                    torch.max(new_times, dim=-1, keepdim=True).values,
                )

            if adapt_dom_rate:
                dynamic_dom_rates[:, k] = (sample_intensities["total_intensity"].max(dim=1).values*100).squeeze(-1)
                k = (k+1) % dynamic_dom_rates.shape[1]  # only keep dyn_dom_buffer num of dom_rates in the buffer
                dominating_rate = torch.max(dynamic_dom_rates, dim=1, keepdim=True).values

            # print(last_time.shape, new_times.shape, proposal_batch_size, dominating_rate.shape)
            new_times = last_time + dist.sample(sample_shape=(new_times.shape[0], proposal_batch_size)).cumsum(dim=-1)/dominating_rate

        if timestamps.shape[0] > 0:  # On the chance that we hit a break after running out of samples within the loop
            if stop_for_marks:
                if (~sample_hasnt_hit_stop_marks).any():
                    resulting_times.append(timestamps[~sample_hasnt_hit_stop_marks, sample_lens-1])
                    resulting_marks.append(marks[~sample_hasnt_hit_stop_marks, sample_lens-1, ...])
                    resulting_states.append(state_values[~sample_hasnt_hit_stop_marks, sample_lens-1, ...])
                    timestamps, marks, state_values = timestamps[sample_hasnt_hit_stop_marks, ...], marks[sample_hasnt_hit_stop_marks, ...], state_values[sample_hasnt_hit_stop_marks, ...]
                if timestamps.shape[0] > 0:
                    resulting_times.append(timestamps[:, -1])
                    resulting_marks.append(marks[:, -1, ...])
                    resulting_states.append(state_values[:, -1])  # This won't be used
            else:
                resulting_times.append(timestamps)
                resulting_marks.append(marks)
                resulting_states.append(state_values)


        assumption_violation = False
        if not adapt_dom_rate:
            for _ in range(5):
                eval_times = torch.rand_like(timestamps).clamp(min=1e-8) * T
                sample_intensities = self.get_intensity(
                    state_values=state_values,
                    state_times=state_times,
                    timestamps=eval_times,
                    marks=None,
                )
                if (sample_intensities["total_intensity"] > dominating_rate).any().item():
                    print("DR: {}".format(dominating_rate))
                    print("IN: {}".format(sample_intensities["total_intensity"].max().item()))
                    assumption_violation = True
                    break

        if assumption_violation:
            print("Violation in sampling assumption occurred. Redoing sample.")
            return None  # self.sample_points(ref_marks, ref_timestamps, ref_marks_bwd, ref_timestamps_bwd, tgt_marks, tgt_timestamps, context_lengths, dominating_rate * 2, T)
        else:
            return resulting_times, resulting_marks, resulting_states  # return a list of tensors containing different lists of sampled sequences



    def compensator(self, a, b, conditional_times, conditional_marks, conditional_states=None, num_int_pts=100,
                    calculate_bounds=False, mark_mask=None):
        '''
            :param mark_mask: torch.Size([batch_size, channel]), val==False -> mark not included
            In estimating the integral, num_events = num_int_pts
        '''
        scalar_bounds = (isinstance(a, (float, int)) and isinstance(b, (float, int))) or ((len(a.shape) == 0) and (len(b.shape) == 0))
        if scalar_bounds:
            assert(a <= b)
        else:
            assert((a <= b).all())

        results = {}
        if conditional_states is None:
            state_dict = self.get_states(conditional_marks, conditional_times)
            conditional_states = state_dict['state_values']
        if scalar_bounds:
            ts = torch.linspace(a, b, num_int_pts).to(next(self.parameters()).device)
            intensity_dict = self.get_intensity(
                state_values=conditional_states,  # state_dict["state_values"],
                state_times=conditional_times,  # state_dict["state_times"],
                timestamps=ts.expand(*conditional_times.shape[:-1], -1),  # unpack all dims except for the last dim
                marks=None,
            )
            vals = intensity_dict['total_intensity']
            ts = ts.expand(*conditional_times.shape[:-1], -1)
        else:
            if len(a.shape) == 0:
                a = a.unsqueeze(0).expand(conditional_times.shape[0])  # torch.Size([batch_size])
            if len(b.shape) == 0:
                b = b.unsqueeze(0).expand(conditional_times.shape[0])

            ts = torch.linspace(0, 1, num_int_pts).unsqueeze(0).to(next(self.parameters()).device)  # torch.Size([1, num_int_pts])
            ts = a.unsqueeze(-1) + ts*(b-a).unsqueeze(-1)  # torch.Size([batch_size, num_int_pts])
            intensity_dict = self.get_intensity(
                state_values=conditional_states,
                state_times=conditional_times,
                timestamps=ts,
                marks=None,
            )
            vals = intensity_dict['total_intensity']  # torch.Size([batch_size, num_int_pts, 1])

        if calculate_bounds:
            delta = (b - a) / (num_int_pts - 1)
            if not scalar_bounds:
                delta = delta.unsqueeze(-1)
            left_pts, right_pts = vals[..., :-1, :], vals[..., 1:, :]  # torch.Size([batch_size, num_int_pts-1, 1])
            upper_lower_pts = torch.stack((left_pts, right_pts), dim=-1)  # torch.Size([batch_size, num_int_pts-1, 1, 2])
            # return values and indices, torch.Size([batch_size, num_int_pts-1, 1])
            upper_vals, lower_vals = upper_lower_pts.max(dim=-1).values, upper_lower_pts.min(dim=-1).values  # take max from both ends on each interval
            results["upper_bound"] = upper_vals.sum(dim=-2) * delta
            results["lower_bound"] = lower_vals.sum(dim=-2) * delta
            results["integral"] = upper_lower_pts.mean(dim=-1).sum(dim=-2) * delta
            if mark_mask != None:
                raise NotImplementedError
        else:
            # weighted compensator of total intensity by marginal probability of an item or a set of items not being included
            if mark_mask != None:
                weights = torch.prod(torch.where(~mark_mask.repeat(vals.shape[0], vals.shape[1], 1), 1 - intensity_dict['item_probability'], 1), dim=-1, keepdim=True)
                vals = vals * (1 - weights)
            results['integral'] = torch.trapezoid(vals.squeeze(-1), x=ts,dim=-1)
        return results


    def compensator_grid(self):
        pass


    @abstractmethod
    def sample_single_set(self, input_dict, idx):
        raise NotImplementedError("Must override sample_single_set")


    @abstractmethod
    def log_likelihood(return_dict, target_marks, right_window, left_window=0.0, mask=None, normalize_by_window=False,
                       normalize_by_events=False, gamma=0.):
        '''
        Computes per-batch log-likelihood from the results of a forward pass (that included a set of sample points).

        :param return_dict: dict_keys(['state_dict', 'intensities', 'sample_intensities'])
        :param target_marks: torch.Size([batch_size, num_events, num_channels])

        :return:
        '''
        raise NotImplementedError("Must override log_likelihood")


    def get_param_groups(self):
        """Returns iterable of dictionaries specifying parameter groups.
        The first dictionary in the return value contains parameters that will be subject to weight decay.
        The second dictionary in the return value contains parameters that will not be subject to weight decay.

        Returns:
            (param_group, param_groups) -- Tuple containing sets of parameters, one of which has weight decay enabled, one of which has it disabled.
        """
        NORMS = (
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.LocalResponseNorm,
        )

        weight_decay_params = {'params': []}
        no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
        for module_ in self.modules():
            # Doesn't make sense to decay weights for a LayerNorm, BatchNorm, etc.
            if isinstance(module_, NORMS):
                no_weight_decay_params['params'].extend([
                    p for p in module_._parameters.values() if p is not None
                ])
            else:
                # Also doesn't make sense to decay biases.
                weight_decay_params['params'].extend([
                    p for n, p in module_._parameters.items() if p is not None and n != 'bias'
                ])
                no_weight_decay_params['params'].extend([
                    p for n, p in module_._parameters.items() if p is not None and n == 'bias'
                ])

        return weight_decay_params, no_weight_decay_params