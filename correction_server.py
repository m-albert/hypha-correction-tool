import os
import numpy as np
import urllib

from imjoy_rpc.hypha import connect_to_server
from tifffile import imread, imwrite
import mrcfile

from skan import Skeleton

def mask_to_lines(mask):
    """
    Convert a mask to a list of lines.
    """

    skel = Skeleton(mask)

    features = []
    for path in range(skel.n_paths):
        coords = skel.path_coordinates(path)[:,::-1].tolist()
        features.append(coords)

    return features


def lines_to_mask(features, shape):
    """
    Convert a list of lines to a mask.
    """

    mask = np.zeros(shape, dtype=np.uint8)

    for line in features:
        for pt in line:
            mask[-int(pt[1]), int(pt[0])] = 1

    return mask


np.random.seed(0)

training_images = []
async def start_server(server_url, path2images, path2labels, save_path):
    """
    Start the server for the correction tool.
    """

    os.makedirs(save_path, exist_ok=True)

    image_basenames = [f.split('_binned')[0]
        for f in os.listdir(path2images) if f.endswith("binned_size_matched.mrc")]

    # Connect to the server link
    server = await connect_to_server({"server_url": server_url})
    token = await server.generate_token()

    def normalize_image(image):
        image = image.astype(float)
        image = (image - image.min()) / (image.max() - image.min()) * 255
        return image.astype(np.uint8)
    
    def get_data_by_index(index=None):

        if index is None:
            # index = 20
            # index = np.random.randint(len(image_basenames)-1)
            index = np.random.randint(20)

        image_basename = image_basenames[index]
        
        image_path = os.path.join(path2images, image_basename + '_binned_size_matched.mrc')
        annotation_pos_path = os.path.join(path2labels, image_basename + '_binned_size_matched_skel_mask_pos.tif')
        annotation_neg_path = os.path.join(path2labels, image_basename + '_binned_size_matched_skel_mask_neg.tif')

        print(image_path)

        image = mrcfile.read(image_path)
        image = normalize_image(image)

        annotation_pos = imread(annotation_pos_path)
        annotation_neg = imread(annotation_neg_path)

        annotation_pos_features = mask_to_lines(annotation_pos)
        annotation_neg_features = mask_to_lines(annotation_neg)

        annotation_pos_features

        return image, annotation_pos_features, annotation_neg_features, image_basename

    
    def get_feature_class_and_id_from_features_lists(feature, features_lists):
        print(feature)
        if feature is None:
            return -1, -1
        feature = feature['geometry']['coordinates']
        features_lists = [fl['features'] for fl in features_lists]
        features_lists = [[f['geometry']['coordinates'] for f in fl] for fl in features_lists]
        for i, features in enumerate(features_lists):
            if feature in features:
                print(i, features.index(feature))
                return i, features.index(feature)
        raise(ValueError("Feature not found"))


    def save_correction(annotation_pos_features, annotation_neg_features, image_basename, image_shape):
    # def save_correction(annotation_pos_features, annotation_neg_features, image_basename):

        print('Saving correction')
        # print(image_basename, image_shape, annotation_pos_features)#, annotation_neg_features)
        annotation_pos_out_path = os.path.join(save_path, image_basename + '_binned_size_matched_skel_mask_pos.tif')
        annotation_neg_out_path = os.path.join(save_path, image_basename + '_binned_size_matched_skel_mask_neg.tif')

        for path, features in zip([annotation_pos_out_path, annotation_neg_out_path], [annotation_pos_features, annotation_neg_features]):
            features = [f['geometry']['coordinates'] for f in features['features']]
            mask = lines_to_mask(features, image_shape)
            imwrite(path, mask)
        
        
    svc = await server.register_service({
        "name": "Correction Tool",
        "id": "correction-tool",
        "config": {
            "visibility": "public"
        },
        "get_data_by_index": get_data_by_index,
        "save_correction": save_correction,
        "get_feature_class_and_id_from_features_lists": get_feature_class_and_id_from_features_lists
        
    })

    plugin_url = "https://raw.githubusercontent.com/m-albert/hypha-correction-tool/main/correction_tool.imjoy.html"

    annotation_sid = svc["id"]
    config_str = f'{{"server_url": "{server_url}", "annotation_service_id": "{annotation_sid}", "token": "{token}"}}'
    encoded_config = urllib.parse.quote(
        config_str, safe="/", encoding=None, errors=None
    )

    annotator_url = (
        f"https://imjoy.io/lite?plugin={plugin_url}&config="
        + encoded_config
    )

    print(annotator_url)

    return server