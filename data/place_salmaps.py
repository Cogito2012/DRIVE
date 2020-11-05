import os
import shutil

def process_subset(data_root, subset):
    coord_path = os.path.join(data_root, subset, 'coordinate')
    for accid in sorted(os.listdir(coord_path)):
        coord_file_path = os.path.join(coord_path, accid)
        for filename in sorted(os.listdir(coord_file_path)):
            vid = filename.split('_')[0]
            name_required = 'maps_%s_%s.avi'%(accid, vid)
            print('processing the video file: %s'%(name_required))
            # check if requried salmap video file exists!
            salmap_video_src = os.path.join(data_root, 'salmaps', name_required)
            if not os.path.exists(salmap_video_src):
                print('salmap video file does not exist! use focus map instead! %s'%(salmap_video_src))
                salmap_video_src = os.path.join(data_root, subset, 'focus_videos', accid, vid + '.avi')
                assert os.path.exists(salmap_video_src), 'video file does not exist! %s'%(salmap_video_src)
            # create destination folder
            salmap_dst_path = os.path.join(data_root, subset, 'salmap_videos', accid)
            if not os.path.exists(salmap_dst_path):
                os.makedirs(salmap_dst_path)
            # copy file
            shutil.copyfile(salmap_video_src, os.path.join(salmap_dst_path, vid + '.avi'))


if __name__ == "__main__":
    raw_data_path = '/ssd/data/DADA-2000'
    process_subset(raw_data_path, 'training')
    process_subset(raw_data_path, 'validation')
    process_subset(raw_data_path, 'testing')
