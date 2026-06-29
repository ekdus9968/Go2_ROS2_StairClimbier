import os
from glob import glob
from setuptools import setup

package_name = 'high_level_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jen',
    description='State machine and follow controller',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'state_machine_node = high_level_planner.state_machine_node:main',
            'simple_follow_node = high_level_planner.simple_follow_node:main',
        ],
    },
)
