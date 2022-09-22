import argparse
import os
import h5py
import categoryMapping
import numpy as np
import cv2
import json


MIN_AREA = 110

def main(ann_id, image_id):
    '''
    Builds COCO/LVIS compatible JSON annotations from Hypersim semantic dataset.
    '''

    # This script expects the following structure:
    tree_structure = """
    root (--dir)
    ├── /semantic_instance/
    │   └── ...
    │       └── ...semantic_instance.hdf5
    └── /instance/
        └── ...
            └── ...semantic.hdf5
    """

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--dir', metavar='I', type=str, required=True,
        help='the root directory where your hypersim "semantic_instance" and "semantic" folder with hdf5 files are stored')

    parser.add_argument('--out', metavar='O', type=str, default="hypersim",
        help='the name of the output file. Default: "annotations')

    args = parser.parse_args()

    print("Searching Hypersim semantic.hdf5 and semantic_instance.hdf5 files. Make sure that these files follow this structure:\n", tree_structure)
    
    filepaths = {}
    counter = 0
    for (root,dirs,files) in os.walk(args.dir, topdown=True):
        for file in files:
            #append the file name to the list
            if file.endswith(".hdf5") and "semantic_instance" in root:
                instance_path = os.path.join(root,file)
                semantic_path = instance_path.replace("semantic_instance", "semantic")
                filepaths[instance_path] = semantic_path
    
    total_files = len(filepaths)
    print("Found", total_files, "files ending with 'hdf5'. Processing files...\n")

    annotation_list = []
    image_list = []

    counter = 0
    for f in filepaths:
        semantic_instance_map = load_hdf5_file(f)
        semantic_map = load_hdf5_file(filepaths[f])
        
        # The semantic_instance maps have arbitrary individual id's, but we need id's that correspond to NYU40 label ids.
        # So for each semantic_instance's first pixel, we take the id from the same pixel from the corresponding semantic file.
        label_map = "moin"
        
        anns, ann_id = segmentationToCocoResult(semantic_instance_map, semantic_map, filepaths[f], ann_id, image_id)
        
        imgs = {}
        imgs["id"] = image_id
        imgs["file_name"] = str(image_id).zfill(12) + ".jpg"
        imgs["dataset"] = "hypersim"
        imgs["height"] = semantic_instance_map.shape[0]
        imgs["width"] = semantic_instance_map.shape[1]
        imgs["tonemap_url"] = ((f.replace("semantic_instance", "tonemap")).replace("geometry_hdf5", "final_preview")).replace("hdf5", "jpg")
        
        image_id += 1
        for a in anns:
            annotation_list.append(a)
        image_list.append(imgs)
        
        counter += 1
        print(round(counter/(total_files/100), 2), "%\tProcessed", counter, "out of", total_files, "files.")
    
    hypersim = {}
    hypersim["annotations"] = annotation_list
    hypersim["images"] = image_list
    
    print("Processing finished. Writing to JSON.")
    
    with open(args.out + '.json', 'w') as fp:
        json.dump(hypersim, fp)
    
    print("Saved to", args.out + '.json')


def load_hdf5_file(path):
    f = h5py.File(path, 'r')
    return f['dataset'][:]


def npArrToBbox(np_arr):
    rows = np.any(np_arr, axis=1)
    cols = np.any(np_arr, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return [float(cmin), float(rmin), float(cmax-cmin), float(rmax-rmin)]

# Adapted from https://github.com/nightrome/cocostuffapi/blob/master/PythonAPI/pycocotools/cocostuffhelper.py
def segmentationToCocoMask(semantic_instance_map, semantic_map, instance_id, filename):
    '''
    Encodes a segmentation mask using the Mask API.
    :param semantic_instance_map: [h x w] instance segmentation map that indicates the instance number of each pixel
    :param semantic_map: [h x w] segmentation map that indicates the label class of each pixel
    :param instance_id: target instance number
    :return: Rs - the encoded label mask for label 'instance_id'
    '''
    segment = semantic_instance_map != instance_id

    s_mask = np.ma.masked_where(segment, semantic_map)
    
    mask_labels, counts = np.unique(s_mask, return_counts=True)
    # Sort labels by most common in case we get overlapping labels.
    mask_labels = mask_labels[np.argsort(-counts)]
    
    if len(mask_labels) > 2:
        print("Warning: Multiple labels detected: ", mask_labels, "Choosing most common label.")

    for l in mask_labels:
        # Ignore the mask value (--)
        if type(l) == np.ma.core.MaskedConstant:
            pass
        # Make sure class id is within NYU40 label range 1-40.
        elif type(l) == np.int16 and l >= 1 and l <= 40):
            # Map hypersim label id to FACSED label id
            label = categoryMapping.hypersim_cid_to_FACSED_cid_map[l]
            # Invert segment
            segment = np.invert(segment).astype('uint8')
            area = float((segment > 0.0).sum())
            if area > MIN_AREA:
                bbox = npArrToBbox(segment)
                segmentation, contours = polygonFromMask(segment, True)
                if len(segmentation) > 0:
                    return "Success", segmentation, area, bbox, label, contours
            else:
                print("Info: Skipping segment due to small size.")
                return "Error", None, None, None, None, None
    
    print("Warning: Segment will not be stored due to invalid polygon or missing label.")
    print("filename:", filename)
    print("instance_id:", instance_id)
    print("mask_labels:", mask_labels)
    return "Error", None, None, None, None, None

def segmentationToCocoResult(semantic_instance_map, semantic_map, filename, ann_id, image_id):
    '''
    Convert a segmentation map to COCO stuff segmentation result format.
    :param semantic_instance_map: [h x w] segmentation map that indicates the label of each pixel
    :param imgId: the id of the COCO image (last part of the file name)
    :param stuffStartId: (optional) index where stuff labels start
    :return: anns    - a list of dicts for each label in this image
       .category_id  - the id of the stuff class of this annotation
       .segmentation - the RLE encoded segmentation of this class
    '''

    # Hypersim instance segmentations do not store floor and wall segments. Instead, saving them as -1.
    # We replace that area of the instance map with the same area from the semantic map, where walls and floors are stored as 1 and 2.
    # This means there will always only be one instance of wall and floor respectively, even if they're occluded.
    # Areas that are labeled -1 in the semantic map are considered unlabeled and therefore must be ignored.
    s_mask = (semantic_map == 1) | (semantic_map == 2)

    # We store walls and floors in id's -100 or lower to avoid overlaps.
    semantic_instance_map[s_mask] = -100 - semantic_map[s_mask]

    shape = semantic_instance_map.shape
    if len(shape) != 2:
        raise Exception(('Error: Image has %d instead of 2 channels! Most likely you '
        'provided an RGB image instead of an indexed image (with or without color palette).') % len(shape))
    [h, w] = shape
    assert h > 0 and w > 0

    labelsAll = np.unique(semantic_instance_map)
    labelsAll = labelsAll[labelsAll != -1]

    anns = []

    for instance_id in labelsAll:

        # Create mask and encode it
        status, segmentation, area, bbox, label, contours = segmentationToCocoMask(semantic_instance_map, semantic_map, instance_id, filename)
        if status == "Error":
            break
        
        # Create annotation data and add it to the list
        anndata = {}
        anndata['bbox'] = bbox
        anndata['category_id'] = int(label)
        anndata['segmentation'] = segmentation
        anndata['area'] = float(area)
        # TODO
        anndata["id"] = ann_id
        anndata["image_id"] = image_id
        # anndata['contours'] = contours
        anns.append(anndata)
        ann_id += 1
    return anns, ann_id


def polygonFromMask(maskedArr, approximate=True):
    # adapted from https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
  
    if approximate:
        approx = []
        for c in contours:
            epsilon = 0.001 * cv2.arcLength(c, True)
            approx.append(cv2.approxPolyDP(c, epsilon, True))
        contours = approx
  
    segmentation = []
    valid_poly = 0
    for contour in contours:
    # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.astype(float).flatten().tolist())
            valid_poly += 1
    if valid_poly == 0:
        print("Warning: Invalid poly. Segment won't be stored.")
        return [], []
    return segmentation, contours


if __name__ == "__main__":
    # The start ID - in this case one more than the last LVIS ID
    ann_id = 1270142
    image_id = 1000001
    main(ann_id, image_id)
