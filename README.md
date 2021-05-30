# DSI Notes and Study Guide
Started this midway through week 2 but I'm sure it's gonna be helpful if I keep up with it.
 
## Object Oriented Programing Notes:
 
## Numpy Notes:
- Here are my notes while working through the Numpy assignment for the second time. Will update with review from Pre-Course later down the line.
- [**Numpy Docs**](https://numpy.org/doc/1.20/reference/index.html)
- [**Completed Assignment**](https://github.com/onewindspirit/numpy)
#### **Basics**
- `import numpy as np`
- `np.array()` is like the fundamental building block
#### **Numpy Arrays** appear listlike and may actions you can take on lists work for arrays, but arrays differ in that:
- Numpy arrays can hold one and only one type of data.
- Numpy arrays are super efficient both in terms of memory footprint and computational efficiency.
- Numpy arrays have a size, and the size cannot be changed.
- Numpy arrays have a shape, which allows them to be multi-dimensional (examples forthcoming).
#### **Indexing Numpy Arrays**
- You can select elements from the array with:
  - `array["start":"end":"step size"]`
  - Zero indexed
  - if you don't care about any properties left of the ones you do care about, you can just leave them blank, leaving the colons
      - Like: `[::-1]`
  - You can reverse the index by setting the step size to a negative
- **Numpy Arrays** can be nested, creating **matrices** like:
  - ``np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])``
  - You can call specific items from the nested arrays with:
      - `array["index in array","index in nested array"...]`
      - You can sort of begin to think of this as "rows" and "columns" of a matrix (though this won't always be an appropriate metaphor)
  - Indexing extends to nested arrays:
      - Example: `x_array[:2, :]`
  - Surgical replacement of values at index like:
      - `x_array[:2, :2] = np.mean(x_array)`
- You can index a Numpy Array with another array (Fancy or Advanced Indexing):
  - Example:
  ```
  colors = np.array(['red', 'blue'])
  idx = np.array([0, 0, 1, 1, 0, 0, 1, 1])
  colors[idx]
  ```
- :bell:**I need to pay close attention to the order I'm putting these things in, it gets confusing and it's not intuitive for me so I probably won't index right the first few tries.**:bell:
#### **Broadcasting**
Operating on an array with another thing:
- Another array
- A single value (called scalar)
- A list (numpy converts this into an array automatically)
- Iterating with comparison objects (>,<,==,etc.) is **very useful** when combined with Boolean Indexing
- Have to use **operator symbols** (&,| rather than and, or)
- You can broadcast an array to another array but they must have complimentary shapes
#### **Boolean Indexing**
This is when stuff starts to click. You can use an array of boolean values to select what elements of another array you return. Moreover, you can combine boolean indexing with broadcasting (especially in the cases where you're using comparison operations, mentioned above) to get to some cool places. This is really where the true nature and utility of numpy arrays seems to come to fruition; even though it's still not the first place my mind goes to when given a list of directions, it seems like I need to develop an instinct for when I need to  use these three concepts in tandem.
:bell:**I need to get the hang of nesting indexing and indexing in general**:bell:
- An example of using Boolean Indexing combined with Broadcasting:
  - Return only the rows where the value in the first column is bigger than five:
  ```
  array[array[:, 0] > 5, :]
  ```
#### **Creating Arrays From Scratch**
There's a couple three ways to make arrays in Numpy:
-  **`np.zeros`** creates an array of zeros:
`np.zeros(10)`, `np.zeros((5, 3))`, etc. gives you an idea about how to define the shape of the array
- **`np.ones`** creates an array of ones
  - can be multiplied with a scalar value to create an array filled with those values
  - `np.ones(3,3) * 5` will create a 3x3 array full of 5's
- You can also fill an array with custom values using `np.fill` but this one is harder to use and does the same thing as what i explained before so I'm ignoring it for now
- **`np.linspace()`** creates an equally spaced grid of numbers between two endpoints. Which is extremely useful for populating stuff quickly and usefully, especially for plotting.
  - Example: `np.linspace(1,5,10)` will create an array of 10 evenly-spaced values between 1 and 5
- **`np.arange`** just like the built-in python function range, but it makes an array. I should commit this one to memory, because it seems super useful but I've been totally oblivious to it up till now
- **`np.random.uniform()`** and **`np.random.normal()`** are quick ways to distribute random numbers in an array, with a similar setup as `np.linspace()`. There are more random variations, but I have yet to really rely on these. :upside_down_face:
- **`np.logspace()`** exists, too but I'm not good enough at math yet to really understand where it'd be utilized appropriately.
#### :exclamation:**Unlike lists, Numpy Arrays cannot be extended, and have a fixed shape and size (which are important to pay attention to because manipulating the size and shape of these will be important in operations between multiple matrices)**:exclamation:
- **`np.array().reshape()`** takes in a tuple of values that will change the shape of the array but, again, the **size of the array cannot be altered**
  - `-1` can be used in **only one** of the reshape values if you don't care about the amount of that value
      - In other words, use `-1` to let numpy pick either the amount of columns or rows for you
  - **Reshaping an array does not make a copy of it**
      - Use `np.array().copy()` to, well, make a copy
#### :rotating_light:**Numpy arrays have 1,2,3 or more dimensions. The amount of integers = the amount of dimensions!**:rotating_light:
#### **Numpy.array Methods**
- Most conventional operations and methods can be called on arrays, though behavior is obviously more complex than it might seem
  - declaring an axis in your method offers slicing/more control
  - `keepdims=True` can be declared in order to keep the shape
- :rotating_light:**The correct method for matrix multiplication (at least in the assessment) is `np.array.dot()`**:rotating_light:
 
## Linear Algebra Notes
 
## Pandas
- *Here are my notes while working through the Pandas assignment for the second time. Will update with review from Pre-Course later down the line.*
- **Pandas** is a library for Python that provides data structure and analysis capabilities akin to SQL
- [**Pandas Docs**](https://pandas.pydata.org/pandas-docs/stable/reference/index.html)
- [**Pandas Cheat Sheet**](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [**Link to Completed Assignment**](https://github.com/onewindspirit/pandas)
- The most basic datatype in Pandas is the **`Series`**: Similar to a numpy array but with added context
  - `Series` datatypes contain an **Index** and a second column of **Values** similar to the values in an array
  -The **index** can be changed from the default zero-indexed range to anything, which provides a key:value relationship similar to a dictionary
- **df.iloc** is positionally based. This indexer accepts integers and integer slices, and essentially treats the data frame as if it were a simple matrix.
- **df.loc** is label based. This indexer works with row and column indices / labels.
- When **loading data** with `pd.read_csv('data',sep=',')` pay close attention to the separator character
- **`pd.to_datetime()`** is extremely useful in **Exploratory Data Analysis** for converting non-standard date and time values into something more uniform and easily read.
- **`pd.DataFrame.groupby()`** doesn't return what I expect it to. You need to add some sort of function or operation at the end in order to get a new dataframe out of it.
 
## Matplotlib Notes
 
## Probability Distributions
- [**Scipy Documentation**](https://docs.scipy.org/doc/scipy/reference/)
- [**Link to Finished Assignment**](https://github.com/onewindspirit/probability-distributions)
### **Lecture Notes**
- **Random variable**: an object that can be used to generate numbers, in a way that valid probabilistic statements about the generated numbers can be made.
- **Distribution**: a pattern of probabilities of a random variable
  - **Discrete Distribution**: usually integers or counts of events.
  - **Continuous Distribution**: usually a range of values representing a measurement of some kind
- **Discrete Functions**:
  - **Probability Mass Function (PMF)**: calculates the relative likelihood of a random variable equaling *t*
  - **Cumulative Distribution Function (CDF)**: enumerates through values that are less than or equal to the variable *t*
     - Can think of this like a compound PMF
     - When graphed, resembles a stepped curve that jumps up at every increment
     - if you need to calculate the probability of values **greater than or equal to** *t*, you can take **1 - cdf**
  - **The "Main" Discrete Distributions:**
     - [**Uniform Distribution**](https://en.wikipedia.org/wiki/Discrete_uniform_distribution): Most familiar discrete distribution. Describes a situation with a finite number of outcomes where each outcome is as likely as another, like a **fair dice roll**.
     - [**Bernoulli Distribution**](https://en.wikipedia.org/wiki/Bernoulli_distribution): Simplest discrete distribution. It's just the model of a single coin, fair or unfair.
        - Signs of a Bernoulli Distribution:
           - Only 2 possible outcomes, 1 and 0
           - probability of *p* that *X* outputs 1
           - Individual trials are independent. The previous trials have no effect on the next
     - [**Binomial Distribution**](https://en.wikipedia.org/wiki/Binomial_distribution): A **counting** distribution. Models a fair or unfair coin flip, counting how many times the result is 1.
        - Parameters include:
           - *n* amount of trials
           - *p* probability that a single flip results in 1
     - [**Hypergeometric Distribution**](https://en.wikipedia.org/wiki/Hypergeometric_distribution) Another counting distribution. Models a deck of cards of 2 types (red and blue, trap and monster, etc.). The deck is shuffled, some number of cards are drawn, and then *x* numbers of cards of a given category are counted from the draw. This count is **hypergeometrically** distributed.
        - Look for problems with decks of 2 types of cards where multiple cards are drawn at one time.
        - Parameters include:
           - *N* total cards in deck
           - *K* total number of cards of one type
           - *n* size of the drawn hand
     - [**Poisson Distribution**](https://en.wikipedia.org/wiki/Poisson_distribution): Another counting distribution. Models are process where **independent and identically distributed (IID)** random events occur at a *fixed average rate* or *frequency* within a set bounds (commonly a fixed amount of time).
        - Key thing to look for is that the events being described are **IID**
        - Parameters include:
           - *𝜆 (Lambda)*: the average rate at which the events occur
        - :bell:**Common in Poisson distribution that conversion between hours, minutes, seconds is necessary to find the Lambda**:bell:
- **Continuous Distributions:**
  - **Probability Density Function(PDF)**: Since a continuous random variable can output any value, using a PMF won't be possible.
  - Finding the **CDF** for continuous distributions isn't as simple as summing PMF's. We need to *integrate* using calculus.
  - The density function does not tell us the probability that our random variable will assume any specific value, but it does tell us the probability that the output of the random variable will fall into any given range. Again, this connection requires integration.
  - When we want to compute probabilities involving some random quantity, we can either:
     - Evaluate the distribution function.
     - Integrate the density function.
  - **The "Main" Continuous Distributions:**
     - [**Uniform Distribution**](https://en.wikipedia.org/wiki/Continuous_uniform_distribution): Uniform distribution comes in continuous flavor, too. It still describes a set of equally likely outcomes, but in this case any number in a chosen interval is a possible output of the random variable. For example, the position of a raindrop falls on a line segment.
     - [**Normal/Gaussian Distribution**](https://en.wikipedia.org/wiki/Normal_distribution): Mostly important for when we get to **Central Limit Theorem**. Has a distinctive :bell:*bell*:bell: shape.
     - [**Exponential Distribution**](https://en.wikipedia.org/wiki/Exponential_distribution): continuous distribution related to the **Poisson Distribution**. Instead of describing observed events at a given rate observed for a specific amount of time, **Exponential Distribution** models the *amount of time* you have to watch until you observe the first event.
        - Examples:
           - the amount of time you have to wait at a bus stop until a bus arrives
           - the amount of space you have to search before you find a dropped object
        - Parameters include:
           *𝜃 or (theta)*: the reciprocal of the rate at which the events occur.
     - [**Gamma Distribution**](https://en.wikipedia.org/wiki/Gamma_distribution): A more general form of exponential distribution. Describes the amount of time you would have to wait until you observe a given number of events occuring (instead of just a single event in the case of the exponential).
- **Distributions in Scipy**
  - Discrete Distribution Objects:
     ```
     uniform_disc = stats.randint(low=0, high=10) # k = 0, 1, ..., 9
     bernoulli = stats.bernoulli(p=0.4)
     binomial = stats.binom(n=50, p=0.4)
     hypergeom = stats.hypergeom(M=20, n=7, N=12) # non-standard parameters! 
     #M = total cards, n = amount of desired cards in deck, N = size of drawn hand
     poisson = stats.poisson(mu=5) # mu is the same as lambda
     ```
  - Continuous Distribution Objects:
     ```
     uniform_cont = stats.uniform(loc=0, scale=10) # non-standard parameters!
     normal = stats.norm(loc=0.0, scale=1.0) # non-standard parameters!
     exponential = stats.expon(loc=2.0) # non-standard parameters!
     ```
  - Calling these objects allows you to use evaluation methods:
     - ```.cdf()```
     - ```.pmf()```
     - ```.rvs()``` to generate random value samples from the distribution object
- ### :rotating_light:**Finding out what the non-standard parameters are is annoying and might be time consuming!**:rotating_light:
- ### :rotating_light:**Until I'm more familiar with the different distribution methods a good way of figuring out which one to use is to look at the amount of parameters indicated in a given problem.**:rotating_light:
- If two random variables have the same distribution they are **Identically Distributed**.
  - Indicated with *~* like *X ~ Y*
- ### :rotating_light:**Get better at figuring out when they're asking you to find percentiles!**:rotating_light:
 
## Binomial Tests
- [**Link to Finished Assignment**](https://github.com/onewindspirit/binomial-tests)
### **Lecture Notes**
- **Differences between Probability and Statistics**:
  - In **probability** we know the parameters of a distribution and we're stying the data generated by the distribution.
  - In **statistics** we have data generated from a *random variable* and we would  like to infer properties of the distribution.
    - Independent and identically distributed data are important, as they allow us to pool information using data all generated from indistinguishable random variables.
    - We can never know exactly the distribution that generated the data, we can only hope to approximate it.
    - We may be able to quantify the uncertainty in our approximation (this is what much of classical statistics is about).

- A **Binomial Test** is a type of *hypothesis test* involving random variables that can be modeled by the **binomial distribution**
- Process of **Hypothesis Testing**:
  1. State a **scientific question**: answer should be yes or no
  2. State a **null hypothesis**: the skeptical answer to the question
  3. State an **alternative question**: the non-skeptical answer to the question (often the inverse of the null hypothesis)
  4. Create a **model**: The model assumes the null hypothesis is true
  5. Set a **rejection threshold**: decide the likelihood of rejecting the null hypothesis
    - The rejection threshold is something you determine based on the nature of the experiment. The more important the results of the experiment are (how certain you need to be), the smaller the rejection threshold should be.
    - A commonly accepted threshold of α is 0.05 (5%)
  6. Collect your data
  7. Calculate a **p-value**: the probability of finding a result equally or more extreme if the null hypothesis is true
  8. Compare the **p-value** to your **rejection threshold**
- ### :rotating_light: Hypothesis testing and the resulting P-values can never prove the null hypothesis true. You can only ***reject the null hypothesis*** or ***fail to reject the null hypothesis***. :rotating_light:

## Sampling Distributions
- [**Link to Finished Assignment**](https://github.com/onewindspirit/sampling-distributions)
### **Lecture Notes**
- *Review from probability distributions:*
  - When we want to compute probabilities involving some random quantity, we can either:
    - Evaluate the distribution function (for discrete distributions).
    - Integrate the density function (for continuous distribution).
  - Both of these are called the **cumulative distribution function** or **CDF**
  - If two random variables have the same distribution function, we say they are **identically distributed**, and we denote this relationship as:
    - *𝑋∼𝑌*
    - In practice this means any probabilistic statements we make about *𝑋* and *𝑌* have the same answer.
- **Sampling Distributions of Statistics**: Consider each individual data point drawon from our population as the outcome from its own random variable
  - any IID sample cane be thought of as a *sequence of random variables that are independent and identically distributed*
- **Statistic**: function of a random sample ***T(X1,X2...Xn)***
  - AKA something we can compute once we have our random ample taken
    - drawing different random samples will result in different values of the statistic
  -Statistics include:
    - Sample mean
    - Sample maximum
    - etc.
- Sampling theory is about *quantifying* the amount of variation of a sample statistic
- In order to do do this, we must:
  - Draw some number of IID data, aka a **sample** from the **population**
  - Compute the **statistic** using the drawn sample
  - Record the statistic's value into a database
  - repeat the process ad nausium
- Once this process is complete we are left with a collection of variations of our statistic, each computed from a different random sample of our variable
- The distribution of the *statistic* that arises from this process is the ***sampling distrubtion of the statistic***
- **The Bootstrap** is a general procedure that allows one to estimate the *variance* (or the total distribution) of **any sample statistic**.
  - The *empirical distribution* of the sample should be our *best guess* to the distribution of the population from which the sample is drawn from. We can illustrate this by comparing the empirical distrution of sample to the actual population distribution functions. 
  - Essentially, since we cannot repeatedly sample from the population, we must sample from the sample itself
- **Bootstrap sample**: a sample taken with replacement from the given dataset whose size is equal to the dataset itself
  - Each bootstrap sample contains its own sample median
  - The sample median taken are thus an approximation to the **distribution of the sample medians**
- The *bootstrap distribution* of the sample median can then be used to estimate statistics that would otherwise be unapproachable
- The Bootstrap is a tool to *quantify the variation in a statistical estimate* useful for almost any situation. It is not possible without a massive amount of computation, hence its relative obscurity in the history of statistics despite its great utility
- Summarized in the assignment: 
  >The Bootstrap is a common, computationally intensive procedure used to approximate the sampling distribution of any sample statistic (which is probably a point estimate of some unknown population parameter). Its power is derived from its generality: it applies to almost all sample statistics we may care about, while other methods (based primarily on the central limit theorem) only apply to the sample mean.

## Central Limit Theorem
### **Lecture Notes**

## Maximum Likelihood
### **Lecture Notes**
- Tying together data from statistics and distribution from probability
- **Maximum likelihood Estimation** is the tool used to pick the most fitting distribution model from a given set of data
  - **Maximum likelihood estimation** is our tool for tuning our **parameters** to our data
  - Lecture notebook explains how this works "under the hood" using the **maximum likelihood method**
      - Calculated by evaluating the pdf at a given point
- **Math Theory Stuff to explore but isn't necessarily super relevant information to memorize when working with computers:**
  - *Big Pi* multiplication through for loop
  - *Big Sigma* summation through for look
  - *Law of Logs* simplifies the likelihood function
  - In practice (Python), we use ***Gradient Descent***

## Hypothesis Testing
### :rotating_light:***Number 1 Thing to take away from Week 2 according to Alex***:rotating_light:
### Lecture Notes ###
- **Steps:**
  1. Take the standard error
  2. Subtract the mean (gives us Zscore)
  3. Take cdf of a normal distribution
  4.
  5. Decided whether or not to **reject the null hypothesis**
- **When you decide the length of an experiment, you have to continue the experiment for the decide length**
- memorizing the math for these tests isn't necessary, but getting familiar with it will help when deciding what test to use in case studies and life in general
- **Mann-Whitney** is useful because it makes no assumptions about distribution
- **T Test** is expected when normality is expected
- Review StatsWeek youtube videos on this subject (in course outline)
  - Null Hypothesis, hypothesis testing and p-value videos specifically
- **Bonferroni Correction** A correction to our p-value (rejection threshold) to drive out error samples
  - Only important if you're doing a lot of tests
 
## Statistical Power
### Lecture Notes
**Statistical power** is a measure of a test's ability to detect an effect/difference when an effect/difference actually exists.
- **Type 1 Error** is a **False positive**
- **Type 2 Error** is a **False negative**
- **Precision** and **Recall** are the methods you should commit to memory
- [Statistical Power Interactive Visualization](https://rpsychologist.com/d3/nhst/)
- The nature of your experiment will determine what type of error to minimize
  - Trying to reduce the amount of errors by balancing the two is a bad practice
  - Skyler's cancer test example
- Tying it into hypothesis testing:
  1. State null and alt hypotheses (H0 and H1)
  2. Choose a level of significant (alpha) and power (1-beta)
     - Compute number of samples required for desired effect size
     - Collect data
  3. Compute the test statistic
  4. Calculate the p-value
  5. Draw conclusions
     - Either reject H0 in favor of H1 or:
     - Fail to reject H0
#### :bell:**I'm really struggling with this and need to go through all the assignments again before the assessment**:bell:
 
## Docker
### Lecture Notes
- **Docker** is a tool used to simplify containing software/application/project packages
  - A *container* is a bundle of applications,softwares,libraries, etc. all delivered together in order to simplify the distribution and cut down on individual memory and processing costs
  - Alternative to virtual machines and virtualization
 
## Bayesian Statistics
### Lecture Notes
#### :rotating_light:**Bayes Rule (colloquially):The prior probability takes into account everything ever observed, then you get new data and calculate the likelihood of seeing this data based on the prior probability. If something observed is really surprising, that will change the prior probability by an extreme amount.:rotating_light:**
#### **P(A|B)= P(B|A)P(A) / P(B)**
- In Bayesian stats, one's previous experiences, or essentially biases, here called **priors**, influence the probability or findings of the experiment.
 
## Bayesian Testing
### Lecture Notes
- Bayes rule applies to distributions as well as scalars
- Represent our beliefs through distributions
- Sample from the distributions to determine our beliefs
- [List of Conjugate Priors](https://en.wikipedia.org/wiki/Conjugate_prior)

