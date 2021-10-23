import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from losses import get_accum_card_error

font = cv2.FONT_HERSHEY_SIMPLEX
org = (12, 25)
fontScale = 0.5
color = (0, 255, 0)
thickness = 2

points_colours = {'upper_left_face_visible': (0, 0, 255),
                  'upper_right_face_visible': (0, 255, 0),
                  'upper_left_back_visible': (255, 0, 0),
                  'upper_right_back_visible': (0, 255, 255),
                  'lower_left_face_visible': (255, 0, 255),
                  'lower_right_face_visible': (255, 255, 0),
                  'lower_left_back_visible': (50, 128, 255),
                  'lower_right_back_visible': (255, 128, 0)}

points_colours_abbreviations = {'upper_left_face_visible': 'ulf',
                                'upper_right_face_visible': 'urf',
                                'upper_left_back_visible': 'ulb',
                                'upper_right_back_visible': 'urb',
                                'lower_left_face_visible': 'llf',
                                'lower_right_face_visible': 'lrf',
                                'lower_left_back_visible': 'llb',
                                'lower_right_back_visible': 'lrb'}


# TODO: Fix the issue with denormalisation?
def show_learning_sample_random(inputs, values, output_value, output_detections, output_points, mean, std, waitTime=0):
    global points_colours, font, org, fontScale, color, thickness
    idx = np.random.randint(0, inputs.shape[0])
    image = np.transpose(inputs[idx].numpy(), [1, 2, 0]) * std + mean
    image = image[:, :, [2, 1, 0]]
    image = Image.fromarray((image * 255).astype(np.uint8))
    image = np.array(image)

    target_value = values[idx].item()

    value = round(output_value[idx].item(), 4)
    detections = output_detections[idx].numpy()
    points = output_points[idx].numpy().reshape(-1, 2)
    points[:, 0] = points[:, 0] * image.shape[0]
    points[:, 1] = points[:, 1] * image.shape[1]

    for i in range(len(points)):
        if detections[i] > 0.5:
            x = int(points[i, 0])
            y = int(points[i, 1])
            image = cv2.circle(image, (y, x), 3, points_colours[list(points_colours.keys())[i]], thickness=-1)

    image = cv2.putText(image, str(int(value * 52)) + "-" + str(int(target_value * 52)), org, font, fontScale, color,
                        thickness, cv2.LINE_AA)

    cv2.imshow('Sample image', image)
    cv2.waitKey(waitTime)
    # cv2.destroyAllWindows()


# TODO: Fix the issue with denormalisation?
def show_learning_sample_batch(inputs, values, output_value, output_detections, output_points, mean, std, waitTime=0):
    global points_colours, font, org, fontScale, color, thickness

    for idx in range(inputs.shape[0]):
        image = np.transpose(inputs[idx].numpy(), [1, 2, 0])  # RGB?
        image = image[:, :, [2, 1, 0]]  # BGR?
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = np.array(image)

        target_value = values[idx].item()

        value = round(output_value[idx].item(), 4)
        detections = output_detections[idx].numpy()
        points = output_points[idx].numpy().reshape(-1, 2)
        points[:, 0] = points[:, 0] * image.shape[0]
        points[:, 1] = points[:, 1] * image.shape[1]

        for i in range(len(points)):
            if detections[i] > 0.5:
                x = int(points[i, 0])
                y = int(points[i, 1])
                image = cv2.circle(image, (y, x), 3, points_colours[list(points_colours.keys())[i]], thickness=-1)

        image = cv2.putText(image, str(int(value * 52)) + "-" + str(int(target_value * 52)), org, font, fontScale,
                            color, thickness, cv2.LINE_AA)

        cv2.imshow('Validation image', image)
        cv2.waitKey(waitTime)
    # cv2.destroyAllWindows()


# TODO: Fix the issue with denormalisation?
def show_mask_points_sample_batch(inputs, values, output_value, output_detections, output_points, output_masks, mean,
                                  std, waitTime=0):
    global points_colours, font, org, fontScale, color, thickness

    for idx in range(inputs.shape[0]):
        image = np.transpose(inputs[idx].numpy(), [1, 2, 0])  # RGB?
        image = image[:, :, [2, 1, 0]]  # BGR?
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = np.array(image)

        masks = np.transpose(output_masks[idx].numpy(), [1, 2, 0])
        # masks = masks/masks.max(axis=(0,1))
        grid = create_mask_grid(masks)
        cv2.imshow('Validation image - Masks', grid)

        target_value = values[idx].item()

        value = round(output_value[idx].item(), 4)
        detections = output_detections[idx].numpy()
        points = output_points[idx].numpy().reshape(-1, 2)
        points[:, 0] = points[:, 0] * image.shape[0]
        points[:, 1] = points[:, 1] * image.shape[1]

        for i in range(len(points)):
            if detections[i] > 0.5:
                x = int(points[i, 0])
                y = int(points[i, 1])
                image = cv2.circle(image, (y, x), 3, points_colours[list(points_colours.keys())[i]], thickness=-1)

        image = cv2.putText(image, str(int(value * 52)) + "-" + str(int(target_value * 52)), org, font, fontScale,
                            color, thickness, cv2.LINE_AA)

        cv2.imshow('Validation image', image)
        cv2.waitKey(waitTime)


# TODO: Fix the issue with denormalisation?
def show_points_from_mask_sample_batch(inputs, values, output_value, output_detections, output_masks, mean, std,
                                       threshold=0.1, waitTime=0):
    global points_colours, font, org, fontScale, color, thickness

    for idx in range(inputs.shape[0]):
        image = np.transpose(inputs[idx].numpy(), [1, 2, 0])  # RGB?
        image = image[:, :, [2, 1, 0]]  # BGR?
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = np.array(image)

        masks = np.transpose(output_masks[idx].numpy(), [1, 2, 0])
        # masks = masks/masks.max(axis=(0,1))
        grid = create_mask_grid(masks)
        cv2.imshow('Validation image - Masks', grid)

        points = get_points_from_mask(masks)
        points[:, 0] = points[:, 0] * image.shape[0]
        points[:, 1] = points[:, 1] * image.shape[1]

        target_value = values[idx].item()
        out_value = torch.argmax(F.softmax(output_value, dim=1), axis=1)
        value = round(out_value[idx].item(), 4)
        # detections = output_detections[idx].numpy()

        for i in range(len(points)):
            if output_detections[idx, i] >= threshold:
                x = int(points[i, 0])
                y = int(points[i, 1])
                image = cv2.circle(image, (y, x), 3, points_colours[list(points_colours.keys())[i]], thickness=-1)

        # image = cv2.putText(image,
        #                     str(int(value * 52)) + "-" + str(int(target_value * 52)),
        #                     org, font, fontScale,
        #                     color, thickness, cv2.LINE_AA)
        image = cv2.putText(image,
                            str(int(value)) + "-" + str(int(target_value * 52)),
                            org, font, fontScale,
                            color, thickness, cv2.LINE_AA)

        cv2.imshow('Validation image', image)
        cv2.waitKey(waitTime)


def show_mask_only_sample(output_masks, idx, save=True, waitTime=33):
    masks = np.transpose(output_masks[idx].cpu().numpy(), [1, 2, 0])
    grid = create_mask_grid(masks)
    cv2.imshow('Validation image - Masks', grid)
    cv2.waitKey(waitTime)
    if save:
        cv2.imwrite('masks_output.jpg', (np.clip(grid, 0, 1) * 255).astype(np.uint8))


def create_mask_grid(masks, padding=((1, 1), (1, 1)), padding_values=((0.5, 0.5), (0.5, 0.5))):
    global points_colours, font, points_colours_abbreviations

    grid1_stack = []
    for i in range(4):
        mask_i = np.pad(masks[:, :, i], padding, 'constant', constant_values=padding_values)
        mask_i = cv2.cvtColor(mask_i, cv2.COLOR_GRAY2BGR)
        colour = points_colours[list(points_colours.keys())[i]]
        cv2.putText(mask_i, points_colours_abbreviations[list(points_colours.keys())[i]], (35, 8), font, 0.25, colour,
                    thickness=1)
        grid1_stack.append(mask_i)
    grid1 = np.hstack(grid1_stack)

    grid2_stack = []
    for i in range(4, 8):
        mask_i = np.pad(masks[:, :, i], ((1, 1), (1, 1)), 'constant', constant_values=((0.5, 0.5), (0.5, 0.5)))
        mask_i = cv2.cvtColor(mask_i, cv2.COLOR_GRAY2BGR)
        colour = points_colours[list(points_colours.keys())[i]]
        cv2.putText(mask_i, points_colours_abbreviations[list(points_colours.keys())[i]], (35, 8), font, 0.25, colour,
                    thickness=1)
        grid2_stack.append(mask_i)
    grid2 = np.hstack(grid2_stack)
    grid = np.vstack([grid1, grid2])
    return grid


def get_points_from_mask(masks):
    mask_points = []
    mask_shape = masks[:, :, 0].shape
    for i in range(8):
        mask_i = masks[:, :, i]
        coordy, coordx = np.where(mask_i == mask_i.max())
        mask_points.append(np.array([[coordy[0] / mask_shape[0], coordx[0] / mask_shape[1]]]))

    mask_points = np.stack(mask_points, axis=1)
    points = mask_points.reshape(-1, 2)

    return points  # normalised! they still need to be muultiplied by the real shape of the image


def show_model_validation(model, validation_loader, mean, std, waitTime=0):
    running_card_error = 0.0
    batch_size = 0
    for batch_index, batch in enumerate(validation_loader):

        if batch_size == 0:
            batch_size = batch['image'].shape[0]

        inputs = batch['image']
        value = batch['value']

        model.eval()
        with torch.no_grad():
            out_mask, out_detections, out_value = model(inputs)

        show_points_from_mask_sample_batch(inputs, value, out_value, out_detections, out_mask, mean, std,
                                           waitTime=waitTime)
        running_card_error += get_accum_card_error(value, out_value).item()
        cv2.destroyAllWindows()

    card_error_epoch = int(round(running_card_error / (len(validation_loader) * batch_size), 1))
    print('Test - Mean Card error:  {} cards.'.format(card_error_epoch))


def get_train_val_indices(dataset, random_seed, validation_split, shuffle_dataset=True):
    # Splitting the dataset into train and validation
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    return train_indices, val_indices
