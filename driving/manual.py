from duckietown_env_argparser import parser_args

from driving_controller import ManualDrivingController

def script():
    args = parser_args()
    controller = ManualDrivingController(args)
    controller.start()

if __name__ == "__main__":
    script()