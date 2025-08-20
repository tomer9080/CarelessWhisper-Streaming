import torch
import numpy as np
import torch.nn.functional as F

class WhisperDataCollatorWithPadding:
    def __call__(self, features):

        input_ids, labels, dec_input_ids, labels_classes, unique_ids = [], [], [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            labels_classes.append([int(item==50257) for item in f["labels"]])
            unique_ids.append(f.get("u_id", 0))

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        # labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=50257) for lab, lab_len in zip(labels, label_lengths)]
        labels_classes = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels_classes, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257 is eot token id

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "labels_classes": labels_classes,
            "unique_id": unique_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}

        batch["input_ids"] = input_ids

        return batch


def pad_2d_sequences(arrays: list, dim: int = 0, padding_value: int = 0) -> torch.Tensor:
    lens = [array.shape[dim] for array in arrays]
    max_len = max(lens)

    padded_arrays = [F.pad(array, (0, int(dim == 1) * (max_len - array.shape[1]), 0, int(dim == 0) * (max_len - array.shape[0])), mode="constant", value=padding_value) for array in arrays]
    return torch.cat([padded_array[None] for padded_array in padded_arrays]) # adding batch dim and concatanating


class LoRAWhisperDataCollatorWithPadding:
    def __call__(self, features):

        input_ids, labels, dec_input_ids, endpoints = [], [], [], []
        
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            endpoints.append(f["endpoints"])
            
        # make a batch
        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        dec_input_ids = torch.nn.utils.rnn.pad_sequence(dec_input_ids, batch_first=True, padding_value=50257)
        endpoints = torch.nn.utils.rnn.pad_sequence(endpoints, batch_first=True, padding_value=-100)

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "endpoints": endpoints
        }

        batch = {k: v.detach() for k, v in batch.items()}

        batch["input_ids"] = input_ids.squeeze(1)

        return batch

