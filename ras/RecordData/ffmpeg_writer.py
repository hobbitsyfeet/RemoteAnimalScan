import time
import subprocess
import numpy as np
import ffmpeg

# class ffmpeg_writer():
#     def __init__(self, path):
#         self.path = path
#         pass

#     def write_colour(self, frame):
#         width = frame.shape[1]
#         height = frame.shape[0]
        
#         process = (
#             ffmpeg
#                 .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=str(fps=30))
#                 .output(self.path, pix_fmt='yuv420p')
#                 .overwrite_output()
#                 .run_async(pipe_stdin=True)
#                 )
#         process.communicate(input=frame)

class ffmpeg_writer():
    def __init__(self, path, res_colour, res_depth=None, fps=24, video_format="avi", compressed=True):
        print(res_colour)

        self.ffmpeg_bin = 'ffmpeg'
        self.res_colour = res_colour
        self.res_depth = res_depth
        self.video_format = video_format
        self.fps = fps
        self.path = path
        self.write_colour_proc = None
        self.write_depth_proc = None
        self.write_imu_proc = None

        self.keep_drain_stderr = True
        # self.thread = threading.Thread(target=self.drain_stderr())
        # self.thread.start()
        # print("Starting " + self.path)



    # Read from pipe.stdrr for "draining the pipe"
    def drain_stderr(self):
        keep_drain_stderr = True
        while True:
            try:
                stderr_output = self.write_colour_proc.stderr.readline()
                print(stderr_output)
            except:
                pass
                # print("FAILED1")
            try:
                stderr = self.write_depth_proc.stderr.readline()
                print(stderr_output)
            except:
                pass
                # print("FAILED2")
            try:
                self.write_imu_proc.stderr.readline()
                print(stderr_output)
            except:
                pass
                # print("FAILED3")
            

    def write_colour_start(self):
        # print((str(self.res_colour[0])+'x'+str(self.res_colour[1])))
        # self.write_colour_proc = (
        self.write_colour_proc = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(1280, 720))
            .output('out.mp4', vcodec='h264', pix_fmt='nv21', **{'b:v': 2000000})
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        
    def write_depth_start(self):
        # GRAY16BE
        self.write_depth_proc = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='gray16be', s='{}x{}'.format(640, 576))
            .output('out.mp4', vcodec='png', pix_fmt='gray16be', **{'b:v': 2000000})
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def write_colour(self, color_frame):
        self.write_colour_proc.stdin.write(
            color_frame.astype(np.uint8).tobytes()
        )

    def write_depth(self, depth_frame):
        self.write_depth_proc.stdin.write(
            depth_frame.astype(np.uint16).tobytes()
        )
        # stderr_output = self.write_depth_proc.stderr.readline()

    def close_proc(self, proc):
        proc.stdin.close()
        proc.stderr.close()
        proc.wait()

    def close(self):
        # if self.write_colour_proc is not None:
            # self.close_proc(self.write_colour_proc)
            # sys.stdout.flush()
        print("Stopping")
        self.write_colour_proc.stdin.close()
        self.write_colour_proc.wait()
            
        # if self.write_depth_proc is not None:
        #     self.close_proc(self.write_depth_proc)
        # if self.write_imu_proc is not None:
        #     self.close_proc(self.write_imu_proc)

        # self.keep_drain_stderr = False
        # self.thread.join()
    
    # def start(self):



    def write(self, colour, depth=None, imu=None):
        # self.write_colour(colour)
        if depth is not None:
            self.write_depth(depth)
        # time.sleep(1/self.fps)

if __name__ == "__main__":
    print("Hello")

    writer = ffmpeg_writer("test", (420,360))
    writer.write_colour_start()
    a = np.zeros((360, 420, 3), dtype=np.uint8)
    a.fill(255)
    write_colour_proc = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(1280, 720))
            .output('out.mp4', vcodec='h264', pix_fmt='nv21', **{'b:v': 2000000})
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
    for i in range(30):

        # print(i)
        # writer.write_colour_proc(a.tostring())
        write_colour_proc.stdin.write(
            a.astype(np.uint8).tobytes()
        )
        # write_depth_proc.stdin.write(

        # )
        # writer.write_colour_proc.stdin.write(a)
        # time.sleep(1/writer.fps)
    # writer.close()
    # for line in writer.write_colour_proc.stderr:
    #     print(line)
        # writer.write(a)
