import imageio
import numpy as np
import wandb


from torch.utils.tensorboard import SummaryWriter

class ImageLogger():
    def __init__(self, my_project_name, my_wandb_username):
        wandb.login(key=my_wandb_username)
        wandb.init(project=my_project_name)

    def capture_image(env):
        return env.physics.render(camera_id=0, width=128, height=128)

    def save_image(image, filename):
        imageio.imwrite(filename, image)

    def log_image_tensorboard(image, writer, step):
        SummaryWriter.add_image('Environment', image.transpose(2,0,1), global_step=step)

    def log_image_wandb(image, step):
        wandb.log({"Environment": [wandb.Image(image, caption="Step {}".format(step))]})