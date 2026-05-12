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
    description='Vision utilities for AutoPark (RDK X5)',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Grayscale converter → /rear_cam/image_gray for VINS-Fusion
            'grayscale_node = vision.grayscale_node:main',

            # Checkerboard IPM auto-calibrator
            'auto_calibrator = vision.ex_calib:main',

            # ── NEW ──────────────────────────────────────────────────────
            # Software NV12 → BGR8 converter (replaces hobot_codec_republish)
            # Launch one instance per camera via parameters.
            'nv12_to_bgr = vision.nv12_to_bgr_node:main',
        ],
    },
)
