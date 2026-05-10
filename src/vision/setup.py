from setuptools import setup
import os
from glob import glob

package_name = 'vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ddddd',
    maintainer_email='ddddd@todo.todo',
    description='Grayscale conversion for RDK X5',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'grayscale_node = vision.grayscale_node:main',
			'auto_calibrator = vision.ex_calib:main'
        ],
    },
)
