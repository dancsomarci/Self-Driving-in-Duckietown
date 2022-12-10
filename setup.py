import sys

from setuptools import find_packages, setup


def get_version(filename):
    import ast

    version = None
    with open(filename) as f:
        for line in f:
            if line.startswith("__version__"):
                version = ast.parse(line).body[0].value.s
                break
        else:
            raise ValueError("No version found in %r." % filename)
    if version is None:
        raise ValueError(filename)
    return version


version = get_version(filename="gym_duckietown/__init__.py")

install_requires = [
    "gym>=0.17.1",
    "numpy>=1.10.0",
    "pyglet==1.5.11",
    "pyzmq>=16.0.0",
    "opencv-python>=3.4",
    "PyYAML>=3.11",
    "duckietown-world-daffy",
    "PyGeometry-z6",
    "carnivalmirror==0.6.2",
    "zuper-commons-z6",
    "typing_extensions",
    "Pillow",
    "stable_baselines3[extra]",
    "seaborn",
    "sb3_contrib",
]

system_version = tuple(sys.version_info)[:3]

if system_version < (3, 7):
    install_requires.append("dataclasses")


setup(
    name=f"duckietown-gym-{line}",
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
    version=version,
    keywords="duckietown, environment, agent, rl, openaigym, openai-gym, gym",
    include_package_data=True,
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "dt-check-gpu=gym_duckietown.check_hw:main",
        ],
    },
)