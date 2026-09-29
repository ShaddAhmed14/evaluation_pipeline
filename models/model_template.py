import docker
import logging
from pathlib import Path
from abc import ABC, abstractmethod

class ModelTemplate(ABC):
    def __init__(self, docker_client, name: str, docker_image_link: str):
        self.docker_image_link = docker_image_link
        self.docker_client = docker_client
        self.name = name
        self.pull_image()

    def pull_image(self):
        try:
            self.docker_client.images.pull(self.docker_image_link)
            print(f"Successfully pulled Docker image: {self.docker_image_link}")
        except docker.errors.APIError as e:
            logging.error(f"Error pulling Docker image: {e}")
            raise ValueError(f"Failed to pull Docker image: {self.docker_image_link}. Error: {e}")

    def run_container_template(self, input_folder: Path, output_folder: Path, additional_commands: list):
            output_folder = output_folder / self.name
            output_folder.mkdir(parents=True, exist_ok=True)

            command = ['--input', '/input', '--output', '/output'] + additional_commands
            volumes = {
                input_folder: {'bind': '/input', 'mode': 'ro'},
                output_folder: {'bind': '/output', 'mode': 'rw'}
            }
            try:
                logging.info(f"Running Docker container {self.docker_image_link} with command: {' '.join(command)}")
                container = self.docker_client.containers.run(self.docker_image_link, command=command, volumes=volumes, detach=True)
                try: # output logs while the container is running
                    for line in container.logs(stream=True):
                        print(f"[{self.name}] --> {line.strip().decode('utf-8')}")
                    result = container.wait()
                    if result['StatusCode'] != 0:
                        logging.error(f"Docker container {self.docker_image_link} exited with status code {result['StatusCode']}")
                finally:
                    container.remove()

            except docker.errors.ContainerError as e:
                logging.error(f"Error running Docker container {self.docker_image_link}: {e.exit_status} - {e.stderr.decode('utf-8') if e.stderr else 'No stderr output'}")
                raise
            except docker.errors.APIError as e:
                logging.error(f"Docker API error while running container {self.docker_image_link}: {e}")
                raise
            except docker.errors.ImageNotFound as e:
                logging.error(f"Docker image not found: {self.docker_image_link}. Error: {e}")
                raise
            except Exception as e:
                logging.error(f"Unexpected error while running Docker container {self.docker_image_link}: {e}")
                raise

    @abstractmethod
    def run_container(self, input_folder: str, output_folder: str):
        raise NotImplementedError("Subclasses must implement the run_container method.")