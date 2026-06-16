import os
from glob import glob
from setuptools import setup

package_name = 'stair_simulation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # World file
        (os.path.join('share', package_name, 'worlds'),
            glob('worlds/*.world')),
        # Launch file
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jen',
    maintainer_email='kan.seyoung1018@gmail.com',
    description='Gazebo simulation with patient stairs',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)