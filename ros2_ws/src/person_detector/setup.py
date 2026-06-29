import os
from glob import glob
from setuptools import setup

package_name = 'person_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jen',
    description='YOLOv8 person detection',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'person_detector_node = person_detector.person_detector_node:main',
            'fake_detector_node = person_detector.fake_detector_node:main',
        ],
    },
)
