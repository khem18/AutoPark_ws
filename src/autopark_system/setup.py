from setuptools import setup
from glob import glob
import os
package_name = 'autopark_system'
setup(
    name=package_name,
    version='0.2.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'firmware_examples'), glob('firmware_examples/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='OpenAI', maintainer_email='user@example.com',
    description='Integrated autoparking master/control package for RDK X5 + ESP32 boards.',
    license='MIT',
    entry_points={'console_scripts': [
        'autopark_master = autopark_system.autopark_master:main',
        'slot_estimator = autopark_system.slot_estimator:main',
        'motion_executor = autopark_system.motion_executor:main',
        'serial_bridge = autopark_system.serial_bridge:main',
        'perception_bridge = autopark_system.perception_bridge:main',
        'flow_distance_node = autopark_system.flow_distance_node:main',
    ]},
)
