
import os
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
TRANSFORM_EXE = ROOT_DIR + "/Transformation_example_x64/transformation_example.exe"
PLAYBACK = "playback"
CAPTURE = "capture"

import concurrent.futures

def mkv_export_ply(mkv_file, timestamp_in_seconds, output, live=False, readout_for_multithread = False):
    timestamp_ms = timestamp_in_seconds*1000

    export_method = None
    if live:
        export_method = CAPTURE
    else:
        export_method = PLAYBACK

    export_command = TRANSFORM_EXE + " " + export_method + " " + mkv_file + " " + str(timestamp_ms) + " " + output

    if readout_for_multithread:
        return [export_command]

    try:
        subprocess.check_call([TRANSFORM_EXE, export_method, mkv_file, str(timestamp_ms), output])
    except subprocess.CalledProcessError:
        return False
    return True

    


def mkv_export_ply_range(mkv_file, timestamp_start, timestamp_end, step, output_basename, create_folder = True, readout_for_multithread = False):

    cmd_list = []
    output_name = ""
    if output_basename[-4:] == ".ply":
        output_name = output_basename[-4:] 
    else:
        output_name = output_basename
    
    if create_folder:
        if not os.path.isdir(output_basename):
            os.mkdir(output_basename)
        directory, file_name = os.path.split(output_basename)
        output_name = output_name + "/" + file_name

    for timestamp in range(int(timestamp_start*1000), int(timestamp_end*1000), int(step*1000)):
        numbered_output = output_name + "_S_" + str(timestamp) + ".ply"
        print(timestamp/1000,"/",timestamp_end, numbered_output)
        cmd = mkv_export_ply(mkv_file, timestamp/1000, numbered_output, readout_for_multithread=readout_for_multithread)
        if readout_for_multithread:
            cmd_list.extend(cmd)

    return cmd_list

def mkv_export_ply_video(mkv_file, step, output_basename):
    # Note requires exiftool
    pass

def mkv_export_ply_range_multi(mkv_file, timestamp_start, timestamp_end, step, output_basename, create_folder = True):
    cmds = mkv_export_ply_range(mkv_file, timestamp_start, timestamp_end, step, output_basename, create_folder, readout_for_multithread =True)
    
    # Use a ThreadPoolExecutor to execute the commands in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_subprocess, cmd) for cmd in cmds]

    # Wait for all futures to complete
    concurrent.futures.wait(futures)


def run_subprocess(cmd):
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout.decode())

if __name__ is "__main__":
    test_file="R:/duck1.mkv"
    outfile="K:/Github/RemoteAnimalScan/data/DuckSamples/"
    # test_file = "K:/Github/RemoteAnimalScan/ras/RecordData/Open3D_Tools/record.mkv"
    # outfile = "K:/Github/RemoteAnimalScan/ras/RecordData/what"

    mkv_export_ply_range_multi(test_file, 1,40,1, outfile)
