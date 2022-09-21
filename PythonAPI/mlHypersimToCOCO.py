import argparse
import os
import h5py

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--dir', metavar='D', type=str, required=True,
        help='the directory where your hdf5 semantic instance files are stored')

    parser.add_argument('--out', metavar='O', type=str, default="annotations",
        help='the name of the output file. Default: "annotations')

    parser.add_argument('--search-depth', metavar='S', type=int, default=3,
        help='how deep the script should search the directory. Default: 3')

    args = parser.parse_args()


    filelist = []
    for (root,dirs,files) in os.walk(args.dir, topdown=True):
        for file in files:
            #append the file name to the list
            if file.endswith(".hdf5"):
                filelist.append(os.path.join(root,file))
    
    total_files = len(filelist)
    print("Found", total_files, "files ending with 'hdf5'.")

    counter = 0
    for f in filelist:
        segmentation = load_hdf5_file(f)
        counter += 1
        print(int(counter/(total_files/100)), "%\tProcessed", counter, "out of", total_files, "files.")


def load_hdf5_file(path):
    f = h5py.File(path, 'r')
    return f['dataset'][:]

if __name__ == "__main__":
    main()
