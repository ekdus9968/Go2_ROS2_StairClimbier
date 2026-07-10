from setuptools import setup

package_name = 'person_detector_camera'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config',
            ['config/camera_params.yaml']),
        ('share/' + package_name + '/launch',
            ['launch/camera_keypoint.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Camera keypoint detection with YOLOv11-pose for person verification',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_keypoint_node = person_detector_camera.camera_keypoint_node:main',
        ],
    },
)
