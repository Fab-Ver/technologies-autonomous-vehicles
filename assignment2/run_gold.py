import argparse
import glob

def main(): 
    parser = argparse.ArgumentParser(description="Folder containg frontview images.")
    parser.add_argument("path", type=str, help="Search path of the directory containing the images to be processed")
    args = parser.parse_args()

    # Get the list of images
    image_paths = glob.glob(args.path)
    print(f"Found {len(image_paths)} images matching '{args.path}'")

if __name__ == "__main__":
    main()