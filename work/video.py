#!/usr/bin/env python3

import time
import numpy as np
import subprocess as sp

NUM_CHAN = 3

def get_info(path):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames,width,height,r_frame_rate', '-of', 'csv=s=,:p=0', path]
    proc = sp.Popen(cmd, stdin=sp.PIPE,
        stdout=sp.PIPE, stderr=sp.PIPE)
    width, height, fr_str, num_frames = proc.stdout.read().decode("utf-8").split(',')
    frn, frd = fr_str.split('/')
    framerate = float(frn)/float(frd)
    width, height, num_frames, framerate = int(width), int(height), int(num_frames), float(framerate)
    # Close process
    close(proc)
    return num_frames, width, height, framerate

def close(proc):
    if proc is not None and proc.poll() is None:
        proc.stdin.close()
        proc.stdout.close()
        proc.stderr.close()
        terminate(proc, 0.2)
    proc = None

def terminate(proc, timeout=1.0):
    """ Terminate the sub process.
    """
    # Check
    if proc is None:  # pragma: no cover
        return  # no process
    if proc.poll() is not None:
        return  # process already dead
    # Terminate process
    proc.terminate()
    # Wait for it to close (but do not get stuck)
    etime = time.time() + timeout
    while time.time() < etime:
        time.sleep(0.01)
        if proc.poll() is not None:
            break

def load_clip(path, tot_frames, framerate, start_frame, num_frames, width, height):
    # Compute start time
    start_time = str(start_frame / framerate)
    # Prep command
    cmd = ['ffmpeg', "-nostats", "-loglevel", "0"] + ['-ss', start_time] + ['-i', path] + ['-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-vframes', str(num_frames), '-f', 'image2pipe',] + ['-']
    proc = sp.Popen(cmd, stdin=sp.PIPE,
        stdout=sp.PIPE, stderr=sp.PIPE)
    arr = np.fromstring(proc.stdout.read(num_frames*width*height*NUM_CHAN), dtype=np.uint8)
    result = np.fromstring(arr, dtype='uint8')

    try:
        result = result.reshape((num_frames, height, width, NUM_CHAN))
    except ValueError:
        print('Fail path:', path)
        print('extract:', start_frame, num_frames, tot_frames)
        print('Current shape:', result.shape)
        print('dims:', num_frames, height, width, NUM_CHAN)
        print('prod:', np.prod((num_frames, height, width, NUM_CHAN)))
        print(proc.stderr.read())
        raise
    # Close process
    close(proc)
    return result

if __name__=='__main__':
    path = './rtdata/video/vi2LHVep_ip3.mp4'
    tot_frames, width, height, framerate = get_info(path)
    num_frames = 64
    start_frame = 3131
    arr = load_clip(path, tot_frames, framerate, start_frame, num_frames, width, height)
    print(arr.shape)
