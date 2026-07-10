import os
from glob import glob
from setuptools import setup

package_name = 'stair_detector'

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
    description='Stair detection',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'lidar_stair_detector = stair_detector.lidar_stair_detector:main',
            'camera_stair_detector = stair_detector.camera_stair_detector:main',
            'yolo_stair_detector = stair_detector.yolo_run:main',
            'fusion_stair = stair_detector.fusion_stair:main',
            'stair_consensus = stair_detector.stair_consensus:main',
        ],
    },
)
