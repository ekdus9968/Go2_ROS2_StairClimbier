from setuptools import setup

package_name = 'stair_detector_lidar'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config',
            ['config/stair_params.yaml']),
        ('share/' + package_name + '/launch',
            ['launch/stair_detector.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Stair detection using LiDAR',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stair_detector_node = stair_detector_lidar.stair_detector_node:main',
        ],
    },
)