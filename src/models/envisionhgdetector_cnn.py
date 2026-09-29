from .model_template import ModelTemplate

class EnvisionHGDetectorCNN(ModelTemplate):
    def __init__(self, docker_client, name: str, docker_image_link: str = None):
        self.docker_image_link = "zephasus/envisionhgdetector_docker:latest"
        super().__init__(docker_client, name, self.docker_image_link)

    def run_container(self, input_folder: str, output_folder: str):
        additional_commands = ['--model', 'cnn_b']
        self.run_container_template(input_folder, output_folder, additional_commands)