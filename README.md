# Thompson Sampling

This package was developed for the course of DS223 at American University of Armenia
by [Elina Israyelyan](https://www.linkedin.com/in/elina-israyelyan/)

The package is created to implement Thompson Sampling algorithm.

### Features

- Implement Thompson Sampling with Beta Distribution
- Implement Thompson Sampling with Normal Distribution
- Do both dynamic and static visualizations for the distributions' pdf functions.

### Usage

```
import thompson_sampling

model = thompson_sampling.model.NormalDistribution() 

# fitting the model 
model.fit(data) 

# predicting the best reward giving arm
model.predict()
```

For further examples check the `examples/` directory
or visit the documentation [website](https://elina-israyelyan.github.io/thompson-sampling/).

### References

A Tutorial on Thompson Sampling.
Available [here](https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf?fbclid=IwAR0hLu4nwhrxs8w7rItNmK-eMQjT_rFIeiE5qyv1-3-34O0XgzcdTWVU61g) [Accessed 14 May 2022]
.\
Introduction to Thompson Sampling.
Available  [here](https://ieor8100.github.io/mab/Lecture%204.pdf) [Accessed 14 May 2022]. \
Github repo BabyRobot. Available [here](https://github.com/WhatIThinkAbout/BabyRobot/tree/master/Multi_Armed_Bandits)