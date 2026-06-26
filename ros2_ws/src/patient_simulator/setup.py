import os
from glob import glob
from setuptools import setup

package_name = 'patient_simulator'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models', 'patient'),
            glob('models/patient/*')),
        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jen',
    description='Patient model + biped teleop',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'teleop_patient = patient_simulator.teleop_patient:main',
            'gait_generator = patient_simulator.gait_generator:main',
        ],
    },
)
