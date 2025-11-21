import os
from teleai_pipe.pipelines.teleai_mmgen.infer_wan import get_pipeline, inferenece_one_video, save_video

if __name__ == "__main__":

    prompt = "一个小男孩正在室内玩耍。他穿着一件黄色的卡通T恤和深蓝色的短裤，头发短而整齐，眼睛里充满了活力和好奇。他坐在柔软的地毯上，身边散落着各色各样的玩具：一辆红色的小汽车、一架蓝色的积木塔，还有一只毛绒恐龙。他正专注地用积木搭建一座“高楼”，小心翼翼地将一块块积木叠在一起，偶尔还会发出“哇哦”的惊叹声。客厅温馨整洁，墙上贴着一些五彩的儿童画，窗外透进柔和的阳光，为整个空间增添了一份温暖与安静。小男孩脸上的笑容洋溢着纯真的快乐，他那专注又兴奋的神情让人感受到童年的美好与想象力的无限。"
    camera_type = "move_left"
    path_vid_save_camcontrol = './temp/video_camctrl.mp4'

    pipeline = get_pipeline()

    inferenece_one_video(path_vid_save_camcontrol, pipeline, prompt, camera_type, start_image=None)

    debug_ = False
    if debug_:
        # test all camera types
        for camera_type in ["move_right", 
                                "move_up", "move_down", 
                                "move_forward", "move_backward", 
                                "rotate_left", "rotate_right"]:

            path_save = f'./temp/infer2/vid_no_start_{camera_type}.mp4'
            inferenece_one_video(path_save, pipeline, prompt, camera_type, start_image=None)

    print('Done')
