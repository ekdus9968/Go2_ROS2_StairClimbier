from setuptools import setup

package_name = 'person_detector_lidar'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/config', ['config/lidar_params.yaml']),
    ('share/' + package_name + '/launch', ['launch/lidar_clustering.launch.py']),
    ],  
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='LiDAR-based person detection using clustering',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_clustering_node = person_detector_lidar.lidar_clustering_node:main',
        ],
    },
)