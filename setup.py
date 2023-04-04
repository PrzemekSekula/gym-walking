from setuptools import setup

setup(
    name='gym_walking',
    version='0.1',
    description='Gym walking environment',
    url='https://github.com/PrzemekSekula/gym-walking.git',
    author='Przemek Sekula',
    author_email='przemeksekula@gmail.com',
    packages=['gym_walking', 'gym_walking.envs'],
    package_data={
        "gym-waking": [
            "envs/img/*.png",
        ]
    },    
    
    license='MIT License',
    install_requires=['gym', 'pygame'],
)
