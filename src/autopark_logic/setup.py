from setuptools import find_packages, setup

package_name = 'autopark_logic'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ddddd',
    maintainer_email='ddddd@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'lot_detector     = autopark_logic.lot_detector:main',
            'ego_pose_bridge  = autopark_logic.ego_pose_bridge:main',  # ← NEW
            'lot_detector_sim = autopark_logic.lot_detector_sim:main',
            'local_mapper     = autopark_logic.local_mapper:main',
            'vins_preflight_check = autopark_logic.vins_preflight_check:main',
        ],
    },
)