from setuptools import setup

package_name = 'person_tracker'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config',
            ['config/tracker_params.yaml']),
        ('share/' + package_name + '/launch',
            ['launch/tracker.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Particle filter for person tracking',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'particle_filter_node = person_tracker.particle_filter_node:main',
        ],
    },
)