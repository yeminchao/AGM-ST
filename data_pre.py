import numpy as np
from sklearn import preprocessing
import scipy.io as sio


def LGnet_data_pre(
    source_data, target_data, source_label, target_label, data_name, times, patch_size
):

    def padWithZeros(X, margin=2):
        newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset : X.shape[0] + x_offset, y_offset : X.shape[1] + y_offset, :] = X
        return newX

    def Patch(data, height_index, width_index):
        height_slice = slice(height_index, height_index + patch_size)
        width_slice = slice(width_index, width_index + patch_size)
        patch = data[height_slice, width_slice, :]
        return patch

    source_train_data_spatial = []
    target_train_data_spatial = []
    target_test_data_spatial = []

    source_label = source_label.flatten()
    target_label = target_label.flatten()
    source_data_spectral = np.reshape(
        source_data, (source_data.shape[0] * source_data.shape[1], -1)
    )
    target_data_spectral = np.reshape(
        target_data, (target_data.shape[0] * target_data.shape[1], -1)
    )

    class_num = len(np.unique(target_label)) - 1

    source_train_idx = sio.loadmat(
        "./training_sample/" + data_name + "/train_idx" + str(times) + ".mat"
    )["source_train_idx"].flatten()
    target_train_idx = sio.loadmat(
        "./training_sample/" + data_name + "/train_idx" + str(times) + ".mat"
    )["target_train_idx"].flatten()
    LGnet_idx = sio.loadmat(
        "./LG_sample/" + data_name + "/target_LG_idx" + str(times) + ".mat"
    )["target_LG_idx"].flatten()

    target_labeled_num = target_train_idx.shape[0]

    target_test_idx = np.concatenate((LGnet_idx, target_train_idx))

    source_train_data_spectral = source_data_spectral[source_train_idx, :]

    target_train_data_spectral = target_data_spectral[target_train_idx, :]

    target_test_data_spectral = target_data_spectral[target_test_idx, :]

    source_data_padding = padWithZeros(source_data, patch_size // 2)

    target_data_padding = padWithZeros(target_data, patch_size // 2)

    for i in range(0, len(source_train_idx)):
        a = source_train_idx[i] // source_data.shape[1]
        b = source_train_idx[i] % source_data.shape[1]
        image_patch = Patch(source_data_padding, a, b)
        source_train_data_spatial.append(image_patch.transpose(2, 0, 1))

    source_train_data_spatial = np.asarray(source_train_data_spatial)
    source_train_label = source_label[source_train_idx]
    source_train_label = source_train_label - 1

    for i in range(0, len(target_train_idx)):
        a = target_train_idx[i] // target_data.shape[1]
        b = target_train_idx[i] % target_data.shape[1]
        image_patch = Patch(target_data_padding, a, b)
        target_train_data_spatial.append(image_patch.transpose(2, 0, 1))

    target_train_data_spatial = np.asarray(target_train_data_spatial)
    target_train_label = target_label[target_train_idx]
    target_train_label = target_train_label - 1

    for i in range(0, len(target_test_idx)):
        a = target_test_idx[i] // target_data.shape[1]
        b = target_test_idx[i] % target_data.shape[1]
        image_patch = Patch(target_data_padding, a, b)
        target_test_data_spatial.append(image_patch.transpose(2, 0, 1))
    target_test_data_spatial = np.asarray(target_test_data_spatial, dtype="float16")
    target_test_label = target_label[target_test_idx]
    target_test_label = target_test_label - 1

    source_train_scaler = preprocessing.StandardScaler().fit(source_train_data_spectral)
    source_train_data_spectral = source_train_scaler.transform(
        source_train_data_spectral
    )

    target_train_scaler = preprocessing.StandardScaler().fit(target_train_data_spectral)
    target_train_data_spectral = target_train_scaler.transform(
        target_train_data_spectral
    )
    target_test_data_spectral = target_train_scaler.transform(target_test_data_spectral)

    return (
        source_train_data_spectral,
        target_train_data_spectral,
        target_test_data_spectral,
        source_train_data_spatial,
        target_train_data_spatial,
        target_test_data_spatial,
        source_train_label,
        target_train_label,
        target_test_label,
        class_num,
        target_labeled_num,
    )


def CDSSnet_LG_data_pre(
    source_data, target_data, source_label, target_label, data_name, times, patch_size
):

    def padWithZeros(X, margin=2):
        newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset : X.shape[0] + x_offset, y_offset : X.shape[1] + y_offset, :] = X
        return newX

    def Patch(data, height_index, width_index):
        height_slice = slice(height_index, height_index + patch_size)
        width_slice = slice(width_index, width_index + patch_size)
        patch = data[height_slice, width_slice, :]
        return patch

    source_train_data_spatial = []
    target_train_data_spatial = []
    target_test_data_spatial = []

    source_label = source_label.flatten()
    target_label = target_label.flatten()
    source_data_spectal = np.reshape(
        source_data, (source_data.shape[0] * source_data.shape[1], -1)
    )
    target_data_spectral = np.reshape(
        target_data, (target_data.shape[0] * target_data.shape[1], -1)
    )

    class_num = len(np.unique(target_label)) - 1

    source_train_idx = sio.loadmat(
        "./training_sample/" + data_name + "/train_idx" + str(times) + ".mat"
    )["source_train_idx"].flatten()
    target_labeled_idx = sio.loadmat(
        "./training_sample/" + data_name + "/train_idx" + str(times) + ".mat"
    )["target_train_idx"].flatten()
    LGnet_idx = sio.loadmat(
        "./LG_sample/" + data_name + "/target_LG_idx" + str(times) + ".mat"
    )["target_LG_idx"].flatten()

    target_labeled_num = target_labeled_idx.shape[0]

    target_train_idx = np.concatenate((LGnet_idx, target_labeled_idx))

    source_train_data_spectral = source_data_spectal[source_train_idx, :]

    target_train_data_spectral = target_data_spectral[target_train_idx, :]

    zero_idx = np.argwhere(target_label == 0).flatten()
    target_del_idx = np.append(zero_idx, target_train_idx, axis=0)
    target_all_num = target_label.shape[0]
    target_all_idx = np.arange(target_all_num)
    target_test_idx = np.delete(target_all_idx, target_del_idx, axis=0)

    target_test_data_spectral = target_data_spectral[target_test_idx, :]

    source_data_padding = padWithZeros(source_data, patch_size // 2)

    target_data_padding = padWithZeros(target_data, patch_size // 2)

    for i in range(0, len(source_train_idx)):
        a = source_train_idx[i] // source_data.shape[1]
        b = source_train_idx[i] % source_data.shape[1]
        image_patch = Patch(source_data_padding, a, b)
        source_train_data_spatial.append(image_patch.transpose(2, 0, 1))

    source_train_data_spatial = np.asarray(source_train_data_spatial)
    source_train_label = source_label[source_train_idx]
    source_train_label = source_train_label - 1

    for i in range(0, len(target_train_idx)):
        a = target_train_idx[i] // target_data.shape[1]
        b = target_train_idx[i] % target_data.shape[1]
        image_patch = Patch(target_data_padding, a, b)
        target_train_data_spatial.append(image_patch.transpose(2, 0, 1))

    target_train_data_spatial = np.asarray(target_train_data_spatial)
    target_train_label = target_label[target_train_idx]
    target_train_label = target_train_label - 1

    for i in range(0, len(target_test_idx)):
        a = target_test_idx[i] // target_data.shape[1]
        b = target_test_idx[i] % target_data.shape[1]
        image_patch = Patch(target_data_padding, a, b)
        target_test_data_spatial.append(image_patch.transpose(2, 0, 1))
    target_test_data_spatial = np.asarray(target_test_data_spatial, dtype="float16")
    target_test_label = target_label[target_test_idx]
    target_test_label = target_test_label - 1

    source_train_scaler = preprocessing.StandardScaler().fit(source_train_data_spectral)
    source_train_data_spectral = source_train_scaler.transform(
        source_train_data_spectral
    )

    target_train_scaler = preprocessing.StandardScaler().fit(target_train_data_spectral)
    target_train_data_spectral = target_train_scaler.transform(
        target_train_data_spectral
    )
    target_test_data_spectral = target_train_scaler.transform(target_test_data_spectral)

    return (
        source_train_data_spectral,
        target_train_data_spectral,
        target_test_data_spectral,
        source_train_data_spatial,
        target_train_data_spatial,
        target_test_data_spatial,
        source_train_label,
        target_train_label,
        target_test_label,
        class_num,
        target_labeled_num,
    )


def CDSSnet_LG_data_pre_all_test(
    source_data, target_data, source_label, target_label, data_name, times, patch_size
):

    def padWithZeros(X, margin=2):
        newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset : X.shape[0] + x_offset, y_offset : X.shape[1] + y_offset, :] = X
        return newX

    def Patch(data, height_index, width_index):
        height_slice = slice(height_index, height_index + patch_size)
        width_slice = slice(width_index, width_index + patch_size)
        patch = data[height_slice, width_slice, :]
        return patch

    source_train_data_spatial = []
    target_train_data_spatial = []
    target_test_data_spatial = []

    source_label = source_label.flatten()
    target_label = target_label.flatten()
    source_data_spectal = np.reshape(
        source_data, (source_data.shape[0] * source_data.shape[1], -1)
    )
    target_data_spectral = np.reshape(
        target_data, (target_data.shape[0] * target_data.shape[1], -1)
    )

    class_num = len(np.unique(target_label)) - 1

    source_train_idx = sio.loadmat(
        "./training_sample/" + data_name + "/train_idx" + str(times) + ".mat"
    )["source_train_idx"].flatten()
    target_labeled_idx = sio.loadmat(
        "./training_sample/" + data_name + "/train_idx" + str(times) + ".mat"
    )["target_train_idx"].flatten()
    LGnet_idx = sio.loadmat(
        "./LG_sample/" + data_name + "/target_LG_idx" + str(times) + ".mat"
    )["target_LG_idx"].flatten()

    target_labeled_num = target_labeled_idx.shape[0]

    target_train_idx = np.concatenate((LGnet_idx, target_labeled_idx))

    source_train_data_spectral = source_data_spectal[source_train_idx, :]

    target_train_data_spectral = target_data_spectral[target_train_idx, :]

    zero_idx = np.argwhere(target_label == 0).flatten()
    target_all_num = target_label.shape[0]
    target_all_idx = np.arange(target_all_num)
    target_test_idx = np.delete(target_all_idx, zero_idx, axis=0)

    target_test_data_spectral = target_data_spectral[target_test_idx, :]

    source_data_padding = padWithZeros(source_data, patch_size // 2)

    target_data_padding = padWithZeros(target_data, patch_size // 2)

    for i in range(0, len(source_train_idx)):
        a = source_train_idx[i] // source_data.shape[1]
        b = source_train_idx[i] % source_data.shape[1]
        image_patch = Patch(source_data_padding, a, b)
        source_train_data_spatial.append(image_patch.transpose(2, 0, 1))

    source_train_data_spatial = np.asarray(source_train_data_spatial)
    source_train_label = source_label[source_train_idx]
    source_train_label = source_train_label - 1

    for i in range(0, len(target_train_idx)):
        a = target_train_idx[i] // target_data.shape[1]
        b = target_train_idx[i] % target_data.shape[1]
        image_patch = Patch(target_data_padding, a, b)
        target_train_data_spatial.append(image_patch.transpose(2, 0, 1))

    target_train_data_spatial = np.asarray(target_train_data_spatial)
    target_train_label = target_label[target_train_idx]
    target_train_label = target_train_label - 1

    for i in range(0, len(target_test_idx)):
        a = target_test_idx[i] // target_data.shape[1]
        b = target_test_idx[i] % target_data.shape[1]
        image_patch = Patch(target_data_padding, a, b)
        target_test_data_spatial.append(image_patch.transpose(2, 0, 1))
    target_test_data_spatial = np.asarray(target_test_data_spatial, dtype="float16")
    target_test_label = target_label[target_test_idx]
    target_test_label = target_test_label - 1

    source_train_scaler = preprocessing.StandardScaler().fit(source_train_data_spectral)
    source_train_data_spectral = source_train_scaler.transform(
        source_train_data_spectral
    )

    target_train_scaler = preprocessing.StandardScaler().fit(target_train_data_spectral)
    target_train_data_spectral = target_train_scaler.transform(
        target_train_data_spectral
    )
    target_test_data_spectral = target_train_scaler.transform(target_test_data_spectral)

    return (
        source_train_data_spectral,
        target_train_data_spectral,
        target_test_data_spectral,
        source_train_data_spatial,
        target_train_data_spatial,
        target_test_data_spatial,
        source_train_label,
        target_train_label,
        target_test_label,
        class_num,
        target_labeled_num,
    )


def TDSSnet_data_pre(target_data, target_label, data_name, times, patch_size):

    def padWithZeros(X, margin=2):
        newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset : X.shape[0] + x_offset, y_offset : X.shape[1] + y_offset, :] = X
        return newX

    def Patch(data, height_index, width_index):
        height_slice = slice(height_index, height_index + patch_size)
        width_slice = slice(width_index, width_index + patch_size)
        patch = data[height_slice, width_slice, :]
        return patch

    target_train_data_spatial = []
    target_test_data_spatial = []

    target_label = target_label.flatten()
    target_data_spectral = np.reshape(
        target_data, (target_data.shape[0] * target_data.shape[1], -1)
    )

    class_num = len(np.unique(target_label)) - 1

    target_labeled_idx = sio.loadmat(
        "./training_sample/" + data_name + "/train_idx" + str(times) + ".mat"
    )["target_train_idx"].flatten()
    LGnet_idx = sio.loadmat(
        "./LG_sample/" + data_name + "/target_LG_idx" + str(times) + ".mat"
    )["target_LG_idx"].flatten()
    target_labeled_num = target_labeled_idx.shape[0]

    target_train_idx = np.concatenate((LGnet_idx, target_labeled_idx))

    target_train_data_spectral = target_data_spectral[target_train_idx, :]

    zero_idx = np.argwhere(target_label == 0).flatten()
    target_del_idx = np.append(zero_idx, target_train_idx, axis=0)
    target_all_num = target_label.shape[0]
    target_all_idx = np.arange(target_all_num)
    target_test_idx = np.delete(target_all_idx, target_del_idx, axis=0)

    target_test_data_spectral = target_data_spectral[target_test_idx, :]

    target_data_padding = padWithZeros(target_data, patch_size // 2)

    for i in range(0, len(target_train_idx)):
        a = target_train_idx[i] // target_data.shape[1]
        b = target_train_idx[i] % target_data.shape[1]
        image_patch = Patch(target_data_padding, a, b)
        target_train_data_spatial.append(image_patch.transpose(2, 0, 1))

    target_train_data_spatial = np.asarray(target_train_data_spatial)
    target_train_label = target_label[target_train_idx]
    target_train_label = target_train_label - 1

    for i in range(0, len(target_test_idx)):
        a = target_test_idx[i] // target_data.shape[1]
        b = target_test_idx[i] % target_data.shape[1]
        image_patch = Patch(target_data_padding, a, b)
        target_test_data_spatial.append(image_patch.transpose(2, 0, 1))
    target_test_data_spatial = np.asarray(target_test_data_spatial, dtype=np.float16)
    target_test_label = target_label[target_test_idx]
    target_test_label = target_test_label - 1

    target_train_scaler = preprocessing.StandardScaler().fit(target_train_data_spectral)
    target_train_data_spectral = target_train_scaler.transform(
        target_train_data_spectral
    )
    target_test_data_spectral = target_train_scaler.transform(target_test_data_spectral)

    return (
        target_train_data_spectral,
        target_test_data_spectral,
        target_train_data_spatial,
        target_test_data_spatial,
        target_train_label,
        target_test_label,
        class_num,
        target_labeled_num,
    )


def TDSSnet_data_pre_all_test(target_data, target_label, data_name, times, patch_size):

    def padWithZeros(X, margin=2):
        newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset : X.shape[0] + x_offset, y_offset : X.shape[1] + y_offset, :] = X
        return newX

    def Patch(data, height_index, width_index):
        height_slice = slice(height_index, height_index + patch_size)
        width_slice = slice(width_index, width_index + patch_size)
        patch = data[height_slice, width_slice, :]
        return patch

    target_train_data_spatial = []
    target_test_data_spatial = []

    target_label = target_label.flatten()
    target_data_spectral = np.reshape(
        target_data, (target_data.shape[0] * target_data.shape[1], -1)
    )

    class_num = len(np.unique(target_label)) - 1

    target_labeled_idx = sio.loadmat(
        "./training_sample/" + data_name + "/train_idx" + str(times) + ".mat"
    )["target_train_idx"].flatten()
    LGnet_idx = sio.loadmat(
        "./LG_sample/" + data_name + "/target_LG_idx" + str(times) + ".mat"
    )["target_LG_idx"].flatten()

    target_labeled_num = target_labeled_idx.shape[0]

    target_train_idx = np.concatenate((LGnet_idx, target_labeled_idx))

    target_train_data_spectral = target_data_spectral[target_train_idx, :]

    zero_idx = np.argwhere(target_label == 0).flatten()
    target_del_idx = np.append(zero_idx, target_train_idx, axis=0)
    target_all_num = target_label.shape[0]
    target_all_idx = np.arange(target_all_num)
    target_test_idx = np.delete(target_all_idx, zero_idx, axis=0)

    target_test_data_spectral = target_data_spectral[target_test_idx, :]

    target_data_padding = padWithZeros(target_data, patch_size // 2)

    for i in range(0, len(target_train_idx)):
        a = target_train_idx[i] // target_data.shape[1]
        b = target_train_idx[i] % target_data.shape[1]
        image_patch = Patch(target_data_padding, a, b)
        target_train_data_spatial.append(image_patch.transpose(2, 0, 1))

    target_train_data_spatial = np.asarray(target_train_data_spatial)
    target_train_label = target_label[target_train_idx]
    target_train_label = target_train_label - 1

    for i in range(0, len(target_test_idx)):
        a = target_test_idx[i] // target_data.shape[1]
        b = target_test_idx[i] % target_data.shape[1]
        image_patch = Patch(target_data_padding, a, b)
        target_test_data_spatial.append(image_patch.transpose(2, 0, 1))
    target_test_data_spatial = np.asarray(target_test_data_spatial, dtype=np.float16)
    target_test_label = target_label[target_test_idx]
    target_test_label = target_test_label - 1

    target_train_scaler = preprocessing.StandardScaler().fit(target_train_data_spectral)
    target_train_data_spectral = target_train_scaler.transform(
        target_train_data_spectral
    )
    target_test_data_spectral = target_train_scaler.transform(target_test_data_spectral)

    return (
        target_train_data_spectral,
        target_test_data_spectral,
        target_train_data_spatial,
        target_test_data_spatial,
        target_train_label,
        target_test_label,
        class_num,
        target_labeled_num,
    )
