import os
import numpy as np
import urllib
import json

from imjoy_rpc.hypha import connect_to_server
from tifffile import imread, imwrite
import mrcfile

from skan import Skeleton

# def mask_to_lines(mask):
#     """
#     Convert a mask to a list of lines.
#     """

#     skel = Skeleton(mask)

#     features = []
#     for path in range(skel.n_paths):
#         coords = skel.path_coordinates(path)[:,::-1].tolist()
#         features.append(coords)

#     return features


# def lines_to_mask(features, shape):
#     """
#     Convert a list of lines to a mask.
#     """

#     mask = np.zeros(shape, dtype=np.uint8)

#     for line in features:
#         for pt in line:
#             mask[-int(pt[1]), int(pt[0])] = 1

#     return mask

def invert_path_coordinate_axes(list_of_paths):
    """
    Invert the x and y coordinates of a list of paths.
    """

    for path in list_of_paths:
        for pt in path:
            pt[0], pt[1] = pt[1], pt[0]
    
    return list_of_paths


np.random.seed(0)

training_images = []
async def start_server(server_url, path2images):
    """
    Start the server for the correction tool.
    """

    # os.makedirs(save_path, exist_ok=True)

    # image_basenames = [f.split('_binned')[0]
    #     for f in sorted(os.listdir(path2images)) if f.endswith("binned_size_matched.mrc")]

    # recursively find all mrc files in the path2images directory
    image_basenames = []
    for root, dirs, files in os.walk(path2images):
        for file in files:
            if file.endswith("binned_size_matched.mrc"):
                # get full file path
                image_path = os.path.relpath(os.path.join(root, file), path2images)
                image_basenames.append(image_path.split('_binned')[0])

    # image_micrograph_paths = [os.path.join(path2images, os.path.dirname(ibn), 'membrainseg', ibn+'_binned_size_matched_paths.json') for ibn in image_basenames]
    # image_input_annotation_paths = [os.path.join(path2images, os.path.dirname(ibn), 'skeletons', ibn+'_binned_size_matched_paths.json') for ibn in image_basenames]
    # image_output_annotation_paths = [os.path.join(path2images, os.path.dirname(ibn), 'corrections', ibn+'_binned_size_matched_paths_corr.json') for ibn in image_basenames]

    image_micrograph_paths = [os.path.join(path2images, ibn+'_binned_size_matched.mrc') for ibn in image_basenames]
    image_input_annotation_paths = [os.path.join(path2images, ibn+'_binned_size_matched_paths.json') for ibn in image_basenames]
    image_output_annotation_paths = [os.path.join(path2images, ibn+'_binned_size_matched_paths_corr.json') for ibn in image_basenames]

    # print(image_basenames)#, image_micrograph_paths, image_input_annotation_paths, image_output_annotation_paths)

    # Connect to the server link
    server = await connect_to_server({"server_url": server_url})
    token = await server.generate_token()

    def normalize_image(image):
        image = image.astype(float)
        image = (image - image.min()) / (image.max() - image.min()) * 255
        return image.astype(np.uint8)

    def get_data_by_basename(image_basename=None):

        if image_basename is None:
            image_basename = image_basenames[0]

       
        index = image_basenames.index(image_basename)        

        if os.path.exists(image_output_annotation_paths[index]):
            print('Loading saved annotation')
            with open(image_output_annotation_paths[index], 'r') as f:
                annotation = json.load(f)
                annotation_pos_features = annotation['keepp']
                annotation_neg_features = annotation['keepn']
                loaded_saved = True
        else:
            print('Loading original annotation')
            with open(image_input_annotation_paths[index], 'r') as f:
                annotation = json.load(f)
                annotation_pos_features = annotation['keepp']
                annotation_neg_features = annotation['keepn']
                loaded_saved = False

        # print(loaded_saved, annotation_save_path)

        print(annotation.keys())
        if 'category' in annotation.keys():
            print('Category found')
            category = annotation['category']
        else:
            print('Category not found')
            category = "Good"

        annotation_pos_features = invert_path_coordinate_axes(annotation_pos_features)
        annotation_neg_features = invert_path_coordinate_axes(annotation_neg_features)

        # print(image_path)

        print(image_micrograph_paths[index])
        image = mrcfile.read(image_micrograph_paths[index])
        image = normalize_image(image)

        annotation_pos_features

        return image, annotation_pos_features, annotation_neg_features, image_basename, loaded_saved, category

    
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


    def save_correction(annotation_pos_features, annotation_neg_features, image_basename, image_shape, category):

        print('Saving correction')

        index = image_basenames.index(image_basename)

        paths_dict = dict()
        for ifeatures, features in enumerate([annotation_pos_features, annotation_neg_features]):
            features = [f['geometry']['coordinates'] for f in features['features']]
            features = invert_path_coordinate_axes(features)
            paths_dict[['keepp', 'keepn'][ifeatures]] = features

        paths_dict['category'] = category

        if os.path.exists(image_output_annotation_paths[index]):
            print('Overwriting existing annotation')
            os.remove(image_output_annotation_paths[index])

        with open(image_output_annotation_paths[index], 'w') as f:
            json.dump(paths_dict, f)

        
    # def get_widget_node_list_of_basenames():
    #     node_list = [
    #         {'title': f.split('_binned')[0], "isLeaf": True, "isDraggable": False, "isSelected": ifile == 0}
    #         for ifile, f in enumerate(image_basenames)]
    #     return node_list

    def get_widget_node_list_of_basenames():

        """
        Example tree structure:
[
                    {"title": 'Item1', "isLeaf": True, "data": {"my-custom-data": 123}},
                    {"title": 'Item2', "isLeaf": True},
                    {"title": 'Folder1'},
                    {"title": 'Folder2', "isExpanded": True,
                        "children": [
                            {"title": 'Item3', "isLeaf": True},
                            {"title": 'Item4', "isLeaf": True}
                        ]
                    }
                ],
        """

        firstExpanded = False
        node_list = []
        # create a node list representing the directory structure of path2images
        for ign in image_basenames:
            # walk through the directory structure
            curr_node_list = node_list
            cum_els = ''
            for iel, el in enumerate(ign.split('/')):
                cum_els = os.path.join(cum_els, el)
                # if the element is not already in the node list
                if el not in [node['title'] for node in curr_node_list]:
                    # add the element to the node list
                    if iel == len(ign.split('/')) - 1:
                        curr_node_list.append({"title": el, "isLeaf": True, "isSelected": ign == 0, "data": {"image_basename": ign}})
                    else:
                        curr_node_list.append({"title": el, "children": [], "isExpanded": False})
                        firstExpanded = True
                        new_curr_node_list = curr_node_list[-1]['children']
                else:
                    # if the element is already in the node list, get the corresponding node
                    new_curr_node_list = [node for node in curr_node_list if node['title'] == el][0]['children']
                # print(el, curr_node_list, new_curr_node_list)
                curr_node_list = new_curr_node_list
        node_list    

        return node_list        
        
    svc = await server.register_service({
        "name": "Correction Tool",
        "id": "correction-tool",
        "config": {
            "visibility": "public"
        },
        "get_data_by_basename": get_data_by_basename,
        "save_correction": save_correction,
        "get_feature_class_and_id_from_features_lists": get_feature_class_and_id_from_features_lists,
        "get_widget_node_list_of_basenames": get_widget_node_list_of_basenames,

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