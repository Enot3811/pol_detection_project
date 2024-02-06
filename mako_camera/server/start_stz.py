import os


def start_pol_cam():
    # os.system("tmux split-window 'python3 /home/firefly/Documents/camera/dual_cam/CAM_Mako_G508B_Pol.py'")
    os.system("tmux split-window 'python3 /home/pc0/projects/pol_detection_project/mako_camera/server/CAM_Mako_G508B_Pol.py'")

def start_pol_rgb():
    # os.system("tmux split-window 'python3 /home/firefly/Documents/camera/dual_cam/CAM_Mako_G507C_RGB.py'")
    os.system("tmux split-window 'python3 /home/pc0/projects/pol_detection_project/mako_camera/server/CAM_Mako_G507C_RGB.py'")
