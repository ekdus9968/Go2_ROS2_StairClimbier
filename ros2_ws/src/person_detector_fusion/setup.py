from setuptools import setup

package_name = 'person_detector_fusion'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config',
            ['config/fusion_params.yaml']),
        ('share/' + package_name + '/launch',
            ['launch/fusion.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='LiDAR + Camera person detection fusion',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'person_fusion_node = person_detector_fusion.person_fusion_node:main',
        ],
    },
)