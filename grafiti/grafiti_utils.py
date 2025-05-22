import torch


def transform_data_for_grafiti(batch_dict, device):
    x_time = batch_dict["observed_tp"].to(device)
    x_vals = batch_dict["observed_data"].to(device)
    x_mask = batch_dict["observed_mask"].to(device)
    y_time = batch_dict["tp_to_predict"].to(device)
    y_vals = batch_dict["data_to_predict"].to(device)
    y_mask = batch_dict["mask_predicted_data"].to(device)

    x_time = torch.cat([x_time, y_time], axis=1)
    y_time = x_time.clone()
    x_likes = torch.zeros_like(x_mask)
    y_likes = torch.zeros_like(y_mask)

    x_mask = torch.cat([x_mask, y_likes], axis=1)
    y_mask = torch.cat([x_likes, y_mask], axis=1)
    x_vals = torch.cat([x_vals, y_likes], axis=1)
    y_vals = torch.cat([x_likes, y_vals], axis=1)
    return x_time, x_vals, x_mask, y_time, y_vals, y_mask
