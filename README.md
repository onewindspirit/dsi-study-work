# DSI Notes and Study Guide

## :dizzy:***```WEEK ONE```***:dizzy:

## Object Oriented Programing:
### **Lecture Notes**
- **Object-oriented programming (OOP)** is a programming paradigm
based on the concept of objects.
- In Python, objects are data structures that contain **data**, known as
**attributes**; and **procedures**, known as **methods**.
- A **class** is a blueprint that describes the format of an object. It tells us what
*attributes* an object will store , and what *methods* that object will have available. The class deﬁnes how an object is built.
- **Magic Method**: Special methods, indicated by double underscore, that you can use to give ubiquitous functionality of some operators to objects deﬁned by your class.
   - `__init__`
   - `__repr__`
   - `__add__`
   - etc.
- **Inheritance** - When a class is based on another class, building off of the existing class to take advantage of existing behavior, while having additional speciﬁc behavior of its own.
- **Encapsulation** - The practice of hiding the inner workings of our class, and only exposing what is necessary to the outside world. This idea is effectively the same as the idea of abstraction, and allows users of our classes to only care about the what (i.e. what our class can do) and not the how (i.e. how our class does what it does).
- **Polymorphism** - The provision of a single interface to entities of different types. This enables us to use a shared interface for similar classes while at the same time still allowing each class to have its own specialized behavior.

## Numpy:
- [**Numpy Docs**](https://numpy.org/doc/1.20/reference/index.html)
- [**Completed Assignment**](https://github.com/onewindspirit/numpy)
### **Lecture Notes**
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
- [**Completed Assignment**]()
### **Lecture Notes**
- **Scalar**: A quantity that only has magnitude
   - Any single value, pretty much
- Most important topic in applied mathematics:
   - Calc in higher dimensional spaces reduces to applied linear algebra
   - Solving regression problems
   - Ranking web pages in order of importance
   - Dimensionality reduction
   - Topic modeling
- **Vector**: Any array of real numbers
   - Quantity that has a magnitude and direction
   - Adding a constant to a  vector adds the constant to each element
      - ***Scalar***
   - **Length of a vector** can be defined with the **Euclidean norm** or the **Taxicab/Manhattan norm**
      - **Euclidean norm**: n-dimensional Euclidean space
         - Pythagorean theorem
      - **Taxicab/Manhattan norm**: distance one would have to "drive" from origin point to point x in a rectangular grid
   - **Cosine distance**: Cosine of two non-zero vectors can be derived using the *Euclidean dot product formula*
   - a **unit vector** is a vector with a length of 1
   - the **dot product** is used to computer and communicate *the angle between two vectors*
   - the **distance** is the norm of the difference between two vectors
   - **Linear combinations** of vectors with the same size are generally added with the *parallelogram rule*
      - A **linear combination** of a collection of vectors is the vector of the form
         - **A linear combination is formed out of a collection of vectors by multiplying them by various constants, and then adding the results**
- **Matrix**: An array with *n* rows and *p* columns
   - Often *n* = number of observations
   - *p* = number of features
   - Invented in the context of **systems of linear equations**
      - Ubiquitous in applied and computational mathematics
      - **Solutions to systems of linear equations** are the most important things for us to be able to compute
   - **Matrix multiplication**: describes how solutions to systems of linear equations are related to one another
      - The two matrices *must be conformable*: the number of columns of the first matrix must equal the number of rows of the second
      - ```np.dot(X,Y)```: how matrix multiplication is often notated in numpy
   - **Matrix transpose**: Flippin' the rows and columns of a matrix
   - **Column vector**: matrix with *n* rows and 1 column
      - denoted with *x* rather than *X* for matrices of higher dimensions
   - **Identity Matrix**: A matrix with 1's along the diagonal and zeros everywhere else that, when multiplied, returns the same value
      - denoted with *I*
   - **Inverse of a matrix**: Inverse of a square *n x n* matrix *X* is an *n x n* matrix *X^-1* such that ***X^-1=XX^-1=I*** where *I* is the *identity matrix*
      - If such a matrix exists, then *X* is said to be **invertible** or **nonsingular**, otherwise *X* is said to be **noninvertible** or **singular**
      - Only square matrices can be invertible
         - Even many square matrices do not have inverses
      - A matrix *X* has an inverse if and only if no column is a linear combination of the remaining columns
         - This means that the columns are **linear independent**
- **Eigenvectors and Eigenvalues**
   - If there is a matrix (A) that multiplied by vector (v) equals a constant(λ) multiplied by that vector (v) then the vector is called an *eigenvector* and that constant is called an *eigenvalue*
      - ***𝐴𝑣=λ𝑣***

## Pandas
### **Lecture Notes**
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
- [**Link to Completed Assignment**](https://github.com/onewindspirit/matplotlib)
### **Lecture Notes**
- Import with:
   ```
      import numpy as np
   import pandas as pd

   import matplotlib.pyplot as plt
   plt.style.use('tableau-colorblind10')

   ## Sometimes needed in Jupyter Notebook
   %matplotlib inline
   ```
- One of many Python libraries for plotting/visualizing data
   - Matplotlib is the de-facto choice
- Matplotlib has a number of different levels of access:
   1. **plt**: Minimal, fast interface for plots and annotations, low complexity
   2. **OO interface w/ pyplot**: Fine-grained control over figure, axes, etc., medium complexity
   3. **pure OO interface** Emped plots in GUI applications, too high complexity
- The **OO interface** is the most advisable access point because it gives you enough control without too much complexity
- **plt** is perfect for quick access of straightforward data but tends to become more trouble than it's worth once things start to get complex
- There's a ton of methods and settings and class stuff in matplotlib and while there's a lot to keep track of its all about visualizing stuff
- **Pandas** and **Matplotlib** go hand in hand
   - Reading datasets into Pandas and generating plots with Matplotlib is clearly fundamental to the course and data science in general

## :sparkles:***```WEEK TWO```***:sparkles:  

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
     - parameters include:
        - *𝜇* or *Mu* for mean
        - *𝜎* or *Sigma* for standard deviation
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
- In **probability** We know the parameters of a distribution and we're keeping the data generated by the distribution.
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
- **Sampling Distributions of Statistics**: Consider each individual data point drawing from our population as the outcome from its own random variable
- any IID sample can be thought of as a *sequence of random variables that are independent and identically distributed*
- **Statistic**: function of a random sample ***T(X1,X2...Xn)***
- AKA something we can compute once we have our random sample taken
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
- repeat the process ad nauseum
- Once this process is complete we are left with a collection of variations of our statistic, each computed from a different random sample of our variable
- The distribution of the *statistic* that arises from this process is the ***sampling distribution of the statistic***
- **The Bootstrap** is a general procedure that allows one to estimate the *variance* (or the total distribution) of **any sample statistic**.
- The *empirical distribution* of the sample should be our *best guess* to the distribution of the population from which the sample is drawn from. We can illustrate this by comparing the empirical distribution of samples to the actual population distribution functions.
- Essentially, since we cannot repeatedly sample from the population, we must sample from the sample itself
- **Bootstrap sample**: a sample taken with replacement from the given dataset whose size is equal to the dataset itself
- Each bootstrap sample contains its own sample median
- The sample median taken are thus an approximation to the **distribution of the sample medians**
- The *bootstrap distribution* of the sample median can then be used to estimate statistics that would otherwise be unapproachable
- The Bootstrap is a tool to *quantify the variation in a statistical estimate* useful for almost any situation. It is not possible without a massive amount of computation, hence its relative obscurity in the history of statistics despite its great utility
- Summarized in the assignment:
>The Bootstrap is a common, computationally intensive procedure used to approximate the sampling distribution of any sample statistic (which is probably a point estimate of some unknown population parameter). Its power is derived from its generality: it applies to almost all sample statistics we may care about, while other methods (based primarily on the central limit theorem) only apply to the sample mean.

## Central Limit Theorem
- [**Link to Finished Assignment**](https://github.com/onewindspirit/central-limit-theorem)
### **Lecture Notes**
- **The Central Limit Theorem** gives us a way to transform non-normal distributions into normal distributions (under certain circumstances)
- ### :rotating_light: *Use `.ppf()` to determine upper and lower bounds of a distribution! :rotating_light:
  - When plotting this, `.axvline` or `.axhline` can generate vertical or horizontal lines on a plot
  - `.fill_between` can perform a similar visualization
- **More on Normal Distribution**:
  - A normal random variable is represented by *Z* rather than *X*
  - *Φ* or *Phi* notates the CDF for a normal random variable
  - Normal distribution parameters:
     - *𝜇* or *Mu* for mean
     - *𝜎* or *Sigma* for standard deviation
  - Changing *Mu* (mean) translates the normal distribution left and right
  - Changing *sigma* (standard deviation) changes the distribution's horizontal scale (squash and stretch)
- **The Central Limit Theorem** asserts that as we take the mean of larger and larger samples, the distribution of the *sample means* becomes more and more normal
  - Said differently, probabilistic statements about the mean of a large sample can be well approximated by assuming that the distribution of the sample means is a normal distribution with the correct mean and variance.
- :bell:**Note**: The CLT does not mean that, given enough samples, any distribution will become normal:bell:
  - To make CLT work, we have to perform some statistic to our samples, then perform and aggregating function ***with central tendency***
     - **Central Tendency** means that the statistic is concerned with finding some central or typical value for the distribution
     - Most of our examples use mean and sample mean
- **Central Limit Theorem** makes no assumptions about the type of distribution or values of our distribution. It can be anything, and the sample mean will always trend normal
- ***The central limit theorem allows us to make probabilistic statements about the sample mean from any population using the normal distribution.***
- **Standard error** is the standard deviation of the *sampling distribution*
### :rotating_light:I'm still trying to suss out how to detect and figure out the standard error and standard deviation for CLT. I will probably have to refer back to the assignment if/when this question gets asked again:rotating_light:
 
## Maximum Likelihood
- [**Link to Finished Assignment**](https://github.com/onewindspirit/maximum-likelihood)
### **Lecture Notes**
- More often than not we do not know the parameters of a distribution
- Since we are usually given a dataset with little or no context for the distribution, our task is not to get data from a set of parameters but to get the parameters of the data
  - This is, in general, what statistics is
- **Statistical Model**: A collection of random variables, each of which is *hypothesized* to possibly have generated the data. Different random variables in the collection are usually distinguished by parameters.
  - *Fitting a statistical model* is any process (*estimation*) that combines a **model** with the **data**
     - Then uses the data to select ***one and only one*** random variable from the model
     - This often takes the form of **determining the parameters for** ***one and only one*** **random variables in the model**
     - These estimated values of the parameters are called **parameter estimates**
- The goal of **Maximum Likelihood Estimation** is to find a *set of parameters* that *best fit* a model distribution to the available data.
- **Maximum likelihood Estimation** is the tool used to pick the most fitting distribution model from a given set of data
- **Maximum Likelihood Estimation** is our tool for tuning our *parameters* to our data
- ***Steps for walking through the Maximum Likelihood Estimation process***:
  1. Take a look at the data
  2. Try to surmise what kind of distribution is at play
  3. Identify which parameter or parameters are unknown
  4. Create a collection of appropriate random variables to 'fill in' our missing parameter(s). This is our *statistical model*
     - Many of these models are improbable
     - A handful of them are reasonable
     - But **only one** is **most likely**
  5. ***Fit the model to the data***
- *Once we're actually fitting the model to the data, we're actually in MLE territory*:
  1. Write down the model
  2. Write down the density functions (pdf) of all random variables in the model
  3. Write code to compute **log likelihood** of the model given the data
     - Remember that the data is *fixed* and we want to vary the *parameters*
  4. Find the parameters that maximize log-likelihood using an algorithm like *gradient descent*
     - This step is known as **Optimization**
- The steps listed above are important to internalize so that we know what maximum likelihood estimation does and where and why to use it. In practice Python will cut down a lot of these steps.
- **Math Theory Stuff to explore but isn't necessarily super relevant information to memorize when working with computers:**
- *∏* or *Big Pi* multiplication through for loop
- *∑* or *Big Sigma* summation through for look
- *Law of Logs* simplifies the likelihood function

## Hypothesis Testing
### :rotating_light:***Number 1 Thing to take away from Week 2 according to Alex***:rotating_light:
- Refer to official solutions instead of assignments. I couldn't get the tests to work even when directly copying from the solutions, so :upside_down_face:
  - [**Link to Solutions**](https://github.com/GalvanizeDataScience/rfe1_solutions)
### Lecture Notes
- Many hypothesis tests are based on the **CLT**
- The approximate test for a population proportion (the "approximation" of a given distribution using a Normal) is called the **Z Test for a population**
  - A two sample approximate test for population proportions is sometimes called a **two sample z-test**
- In the case that normal distribution is expected, a **T Test** can be used
  - **Welch's t-test**
  - **Student's t-test**
  - Because t-tests only work when the *distribution is normal* It is not usually advised as a good means of hypothesis testing.
     - Insted it is advised to use a test where there are *no distributional assumptions*
        - An example of such tests is the [**Mann-Whitney Signed Rank Test**](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test)
           - Used by Alex Martinez for one of his capstone projects
- **Distribution of P-Values Under the Null Hypothesis**: explored in order to develop a *rejection threshold*
- **Worst Case Long Term False Positive Rate**: In the event that *all our hypotheses are false* we must use a rejection threshold of 𝛼 for their experiments will have, in the long term, a false positive rate of 𝛼
- **Bonferroni Correction** A correction to our p-value (rejection threshold) to drive out error samples
- Only important if you're doing a lot of tests
- **When you decide the length of an experiment, you have to continue the experiment for the decide length**
- memorizing the math for these tests isn't necessary, but getting familiar with it will help when deciding what test to use in case studies and life in general
[**StatQuest video on Hypothesis Test and Null Hypothesis**](https://www.youtube.com/watch?v=0oc49DyA3hU)
 
## Statistical Power
- [**Link to Finished Assignment**](https://github.com/onewindspirit/statistical-power)
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
   2. Choose a level of significance (alpha) and power (1-beta)
      - Compute number of samples required for desired effect size
      - Collect data
   3. Compute the test statistic
   4. Calculate the p-value
   5. Draw conclusions
      - Either reject H0 in favor of H1 or:
      - Fail to reject H0

## Docker
### Lecture Notes
- **Docker** is a tool used to simplify containing software/application/project packages
- A *container* is a bundle of applications,softwares,libraries, etc. all delivered together in order to simplify the distribution and cut down on individual memory and processing costs
- Alternative to virtual machines and virtualization

## Bayesian Statistics
- [**Link to Finished Assignment**](https://github.com/onewindspirit/bayes-intro)
### Lecture Notes
#### :rotating_light:**Bayes Rule (colloquially):The prior probability takes into account everything ever observed, then you get new data and calculate the likelihood of seeing this data based on the prior probability. If something observed is really surprising, that will change the prior probability by an extreme amount.:rotating_light:**
#### **P(A|B)= P(B|A)P(A) / P(B)**
- In Bayesian stats, one's previous experiences, or essentially biases, here called **priors**, influence the probability or findings of the experiment.

## Bayesian Testing
- [**Link to Finished Assignment**](https://github.com/onewindspirit/bayes-testing)
### Lecture Notes
- Bayes rule applies to distributions as well as scalars
- Represent our beliefs through distributions
- Sample from the distributions to determine our beliefs
- [List of Conjugate Priors](https://en.wikipedia.org/wiki/Conjugate_prior)

## :zap:***```WEEK THREE```***:zap:

## Intro to SQL
- [**Link to Finished Assignment**](https://github.com/onewindspirit/sql-intro)
### **Lecture Notes**
- **Relational Database Management System (RDBMS)**:
   - Persistent data storage system
      - survives after the rpocess in which it was created has ended
      - is written to non-volatile storage
   - de facto standard for storing data
   - Examples include: Oracle, MySQL, SQLServer, Postgres
   - Non-relational models are used for some applications like data science
   - Provides the ability to:
      - **Model** relations in data
      - **Query** data and their relations efficiently
      - **Maintain** data consistency and integrity
   - Requires a **Data Model**
      - **Schema** defines the structure of the data
      - **Database** is composed of a number of user-defined **tables**
      - Each **table** will have **columns** (or fields) and **rows** (or records)
      - A column is of a given **data type**
      - A row is an entry in a table with data for each column of the given table
**SQL**:
   - Language used to query relational databases
   - **All RDBMS use SQL** and the syntax and keywords are the same for the most part, across systems
   - **SQL is used to interact** with RDBMS, allowing you to create tables, alter tables, insert records, update records, delete records and query records within and across tables
   - Many non-relational databases usually have a SQL-like interface available
**SQL** can:
   - Create and alter tables in a database
   - Insert, update and delete records in tables
   - :rotating_light:**Query (SELECT) records within or across tables**:rotating_light:
**SQL** queries have 3 main components:
   - ```SELECT```: list of columns to query from table
   - ```FROM```: Specify what table to read from
   - ```WHERE```: What rows to return
- **Aggregators** like ```MAX, SUM, COUNT``` can be used in the ```SELECT``` or similar components
- ```GROUP BY``` calculates aggregate stats for groups of data
   - **Any column that is not an aggregator** ***must*** **be in the** ```GROUP BY ``` **clause**
   - Any column in the ```GROUP BY``` clause must also appear in the ```SELECT``` clause
   - ```WHERE``` sets conditions or filters for the ```SELECT``` clause
   - ```HAVING``` behaves like ```WHERE``` but is used for filtering *after* aggregation (usually placed after the ```GROUP BY``` clause)
   - ```JOIN``` combines multiple tables in the ```SELECT``` clause
      - Functions identically to the join function in Pandas
- **Subqueries** appear necessary when multiple or complex aggregations are desired
- Skylar recommends using ***Temporary Tables*** or ***Create/Drop Table*** instead as they are more explicit and readable methods
- ```VIEW``` and **Window Functions** can also be used in complex, dynamic queries

## Advanced SQL
- [**Link to Finished Assignment**](https://github.com/onewindspirit/sql-advanced)
- [**Psycopg2 Documentation**](https://www.psycopg.org/docs/install.html)
### **Lecture Notes**
- Connecting SQL to python is vital for machine-learning applications
   - SQL-based databases are extremely common in almost all industry environments
   - Can leverage the benefit of SQL's structure and scalability, while maintaining the flexibility of Python
   - Very useful for scaled data pipelines, pre-cleaning, data exploration
   - Allows for dynamic query generation and hence automations
- **Psycopg2**: Python library that allows for connections with **PostgresSQL** databases for query and retrieval of data for analysis
- **General Workflow**:
   1. Establish connection to Postgres database with psycopg2
      - Connections must be established using an existing database, username, database IP/URL, potentially passwords
      - To create a database, you can connect to Postgres using the dbname *postgres* to initialize
   2. Create a ***cursor***
      - A **cursor** is a control structure which enables traversal over the records in a database
      - Executes and fetches data
      - When the cursor points at the resulting output of a query, it can only read each observation once
         - If you choose to see a previously read observation, you *must rerun the query*
      - Can be closed without closing the connection
   3. Use the cursor to execute SQL queries and retrieve data
   4. Commit SQL actions
      - Data changes are not stored until they are committed
      - You can set ```autocommit = True```
      - When connecting directly to a Postgres Server to initiate server level commands, ```autocommit``` must ```= True```
   5. Close the cursor and connection
      - Cursors and connections must be closed using ```.close()``` or else Postgres will lock certain operations
- **Dynamic Queries**: Queries that generate based on context
- :rotating_light:**Beware of SQL Injection!**:rotating_light:
   - SQL injection is a code injection technique that might destroy your database.
   - SQL injection is one of **the most common** web hacking techniques.
   - SQL injection is the placement of malicious code in SQL statements, via web page input.
   - SQL injection usually occurs when you ask a user for input, like their username/userid, and instead of a name/id, the user gives you an SQL statement that you will **unknowingly run on your database**
- :bell:**Key things to keep in mind about psycopg2**:bell:
   - Connections must be established using an existing database, username, database IP/URL and sometimes passwords
   - If you have no databases, you can connect to Postgres using the dbname 'postgres' in order to initialize db commands
   - Data changes are not actually stored until you commit
      - Can be done through ```conn.commit()``` or setting ```autocommit = True```
         - Until committed, all transactions are **only temporarily stored**
   ```Autocommit = True``` is necessary to do database commands like ```CREATE DATABASE```
      - This is because Postgres does not have temporary transactions at the database level
   - If you need to build similar pipelines for other forms of databases, libraries like **PyODBC** operate similarly to psycopg2
   - SQL connection databases utilize **cursors** for data traversal and retrieval
      - Similar to an iterator in Python
   - Cursor operations typically function like:
      1. Execute a query
      2. Fetch rows from query result if it is a ```SELECT``` query
         - Because it is iterative, previously fetched rows can only be fetched again by rerunning the query
      3. Close the query through ```.close()```
   - Cursors and Connections must be closed using ```.close()``` or else Postgres will lock certain operations
- **Relational Database Management Systems (RDBMS):**
   - ***Persistent data storage systems***
      - Schema defines the structure of a table or database
      - Database is composed of a number of user-defined tables
      - Each table has columns (*fields*) and rows (*records*)
      - A column is of a certain data type (such as an integer, text or date)
   - With a new data source, your first task is typically to understand the schema
      - Likely takes tame and conversations with those that gave you access to the database or data
   - **RDBMS and SQL**
      - SQl is the language used to query relational databases
      - **All RDBMS** use SQL and the syntax and keywords are the same (for the most part) across all systems
      - **SQL is used to interact** with RDBMS, allowing you to create and alter tables, update records, delete records, and query records within and across tables
      - Even non-relational databases usually have a SQL-like interface available
- **Database Life Cycle (DBLC)**:
   - Requirements Elicitation -> Database Design -> Implementing and loading -> Operations -> Maintenance
   - --->***Datascience***--->
   - Normalization -> Primary Keys, Foreign Keys, ```CREATE, DROP, INSERT```-> ```SELECT, FROM, WHERE, JOIN, HAVING, DISTINCT, GROUP BY, ORDER BY```
- **Relational Database Management Systems**
   - A **RDBMS** is a type of database where data is stored in multiple related tables.
   - The tables are related through primary and foreign keys.
- **Primary Keys**
   - Every table in a RDBMS has a **primary key** that *uniquely identifies* that row
   - Each entry must have a primary key, and primary keys *cannot repeat* within a table
   - Usually integers, often *GUIDs* or *UUIDs* but that is not always the case
- **Foreign Keys and Table Relationships**
   - A **foreign key** is a column that uniquely identifies a column in another table
   - Usually a foreign key in one table is a primary key in another table
   - Foreign keys are used to join tables
- **Entity Relationship Diagram (ERD)**
   - An ERD represents how each foreign key and primary key join multiple tables
   - When using real production data, these diagrams can take up many pages
- **Database Normalization**
   ***Store each piece of information in exactly one place***
      - Details about a user (address, age) are only stored ones (in a users table)
      - Any other table (eg. purchases) where this data might be relevant, only references the user_id
- **Normal Forms**
   - Normalization is explained through *Normal forms*
      1. *First Normal Form*: Each attribute has one value
      2. *Second Normal Form*: No dependencies on part of a key
      3. *Third Normal Form*: No transitive dependencies
- Some databases are not fully normalized:
   - Data structure is not known
   - Data structure may change
   - Simple queries are important
   - Data will not be changed
   - Storage is cheap
- **Data Science in the DBLC**
   - Data science operations:
      - Querying
      - Aggregating
   - Data science implementation:
      - Identifying, cleaning, pushing external data sources inside a RDBMS
   - Data science design:
      - Recommendation on the model, specs on operations
- **Order of Evaluation of a SQL SELECT STatement:**
   1. ```FROM + JOIN```: first the product of all tables is formed
   2. ```WHERE```: filters rows that do not meet the search condition
   3. ```GROUP BY + (COUNT, SUM, etc)```: the rows are grouped using the columns in the group by clause and the aggregation functions are applied on the grouping
   4. ```HAVING```: Like the ```WHERE``` clause, but can be applied after aggregation
   5. ```SELECT```: the targeted list of columns are evaluated and returned
   6. ```DISTINCT```: duplicate rows are eliminated
   7. ```ORDER BY```: the resulting rows are sorted
- **Transactions**: Groups of database statements
   - Ended with ```commit``` or ```rollback```
   - Default interactive behavior is ```autocommit```
- **Transactions are**:
   - Atomic
   - Consistent
   - Isolated
   - Durable

## MongoDB
- [**Link to Finished Assignment**]()
### **Lecture Notes**
- **Big Data**:
   1. Any amount of data that is larger than what you could store and work with on a single machine
   2. Data that many people around the world need to access simultaneously
   3. Data that is coming in from multiple sources simultaneously
   - ***The Three V's***
      - Volume
      - Velocity
      - Variety/Variability
   - Big Data is often *unstructured* and *unorganized*
- With SQL we always know:
   - What data to expect
   - What type of data to expect
   - Which fields must always exist
   - Which fields may always exist
   - How it will be related to other data
- We use this information to define an up-front *schema*
   - Makes data manageable
   - Optimizes for complicated data operations
- An optimized database is reduced to a *normal form*
   - The data is never *repeated*
   - Instead the data is *referenced* by another table using a *key relationship*
- [Database Normal Forms](https://en.wikipedia.org/wiki/Database_normalization)
   - Normalization can create a lot of tables
   - SQL is an *imperative* language
      - The server is free to implement its own methods to achieve whatever result requested by the user
- SQL Downsides/Limitations:
   - Complex code can't be read top-to-bottom
   - Rigid order of operations
   - Difficult to access and view intermediate results
   - Difficult to restructure code for minor changes
   - Blocks of code are frequently repeated
   - Not object-oriented
   - Small changes in structure or order of operations can have large impact on performance
   - Difficult to implement unit testing
   - Difficult to link to other ML libraries such as *sklearn* and *tensorflow*
- :collision:***Big problem with SQL, main reason for using NoSQL databases***:collision:
   - Requires rigid, pre-defined schema
   - Schema must be developed before data is entered
   - Changes to schema after data has been collected can break existing code and require extensive restructuring
- **No SQL**
   - *No SQL* refers to *Not Only* SQL rather than *no* SQL
   - Advantages:
      - Does not require a schema before data collection
      - New data with different naming or structure can be added without breaking existing schema
      - Lists, sets or dictionaries can be added as fields without requiring normalization
   - Disadvantages:
      - Slower performances
      - Occasionally more complicated rules
      - Requires a lot of energy for analyzing, indexing, organizing data
   - Intended to emulate the best parts of SQL:
      - Searching, filtering, updating, deleting, grouping, aggregating, joining data
- **MongoDB**
   - One of several *document* based NoSQL databases
   - ```pymongo``` is used to interact with MongoDB using Python
   - Document-oriented database
      - Alternative to RDBMS
      - used for storing *semi*-structured data
   - **jSON**-like objects form the data model
      - Rather than RDBMS tables
   - Schema is optional
   - *Sub-optimal for complicated queries*
   - Definies fields at the *document* level rather than the table level
      - Each document can have its own structure
      - Makes NoSQL and MongoDB more flexible but less efficient
   - A MongoDB database is made up of ```collections```
      - Containers for actual stored data
      - ```Collections``` are analogous to ```tables``` but are much more flexible
- **Javascript Object Notation (JSON)**:
   - simple data storage and communication protocol
   - similar to string representation of a Python dictionary
      - has ```key:value``` relationships
   - Can contain many different pieces of data and data types
   - By contrast, a SQL database is more like a pandas dataframe

## Web Scraping
- [**Link to Finished Assignment**]()
### **Lecture Notes**
- Pull down data from the web that does not have a clickable link via the command line or program
- The web is an enormous database of text/image/video training data
   - Be aware of terms of service
- The ***world wide web*** is a global collection of interconnected hypermedia documents hosted on **web servers**
- The ***Internet*** is the global network that connects them (using TCP/IP)
- Think of web servers as islands existing all over the globe
- Think of the internet as a service that provides bridges to connect the islands
- **Uniform Resource Locator (URL)**: Used to specify the location of a document within the world-wide web
   - Contains:
      - **Protocol**: Specifies the means for communicating with the web server
         - typically ```http``` or ```https```
      - **Host**: Points to the name of the web server you want to communicate with
         - Associated with a specific IP address
      - **Port**: Holds additional information used to connect with the host
      - **Path**: Indicates where on the server the file you are requesting lives
- At any point, any person or machine connected to the internet can either be a **server** or a **client**
- **Client**: requesting party
- **Server**: party providing information requested by a client
- **Http Status Codes**:
   - 2xx = success
   - 3xx = redirect
   - 4xx = client-side error
   - 5xx = server-side error
- **Hypertext Markup Language (HTML)**:
   - Majority of web pages are formatted using HTML
   - HTML uses tags to describe different elements of the document
   - **Cascading Style Sheets (CSS)** permit clean separation of content and presentation
      - CSS include *classes*, *IDs* and *Selectors* to identify different elements for styling
   - HTML tags and CSS style classes make great hooks for webscraping
- **General Workflow**:
   1. Find the page you want to scrape info from
   2. Find the element(s) that you want to grab
   3. Use ```inspect element``` to figure out what HTML tag or CSS selectors to use as hooks
   4. Use python to hetch those elements
      - *BeautifulSoup*
      - Pandas
      - Web APIs
      - Pymongo

## AWS
- [**Link to Finished Assignment**]()
### **Lecture Notes**
- **AWS cloud computing benefits:**
   - AWS provides on-demand use of resources
   - No need to build out data centers
   - Easy to create new businesses
   - Only pay for what you use
   - Hand spikes in computational demand
   - Secure, reliable, flexible, scalable, cost-effective
- **AWS core services:**
   - ***Elastic compute cloud (EC2):*** Computers for diverse problems
   - ***Elastic block store (EBS):*** Virtual hard disks to use with EC2
   - ***Simple storage solution (S3):*** Long-term bulk storage
   - ***DynamoDB: NoSQL database***
   - ***And many more!***
- **Simple storage solution (S3):**
   - S3 Bucket:
      - Container for files
      - Logical grouping of files
      - Can contain any arbitrary number of files
      - Max bucket size in AWS is 5 TB
   - S3 Bucket Names:
      - Should be DNS-compliant
      - Must be between 3-63 characters long
      - Can contain lowercase letters, numbers and hyphens only
      - Series of one or more labels
         - Adjacent labels are separated by a ```.```
         - Each label must start and end with a lowercase letter or a number
      - Bucketnames must NOT be formatted as an IP address
   - Manage buckets in Python using the ```boto3``` library
- **Elastic compute cloud (EC2):**
   - Divided into *AWS regions*
      - Defaults to ```us-east-1 ```
         - Picking another region will require additional configuration
      - Divided further into *Availability Zones*
         - Regions are divided into zones in order to provide *fault tolerance*
         - Availability zones run on physically separate hardware and infrastructure
         - Do not share hardware, generators, etc.
         - Assigned automatically to your EC2
   - You can only download a specific ```key pair``` once when starting an EC2 instance
      - If the ```key pair``` is lost or you need to connect from a different machine you must *generate a new key pair*

## SQL and Dataframes in Spark
- [**Link to Finished Assignment**]()
### **Lecture Notes**
- *Spark* is a tool used for **parallelized computation**
   - Highly efficient distributed operations
   - Runs in memory and on disk
- **Resilient Distributed Datasets (RDD)**:
   - Primary class introduced by Spark
   - **Immutable**
   - **Lazily Evaluated**
   - **Cacheable**
- Spark provides many *transformation functions*
   - By programming these functions, one constructs a **Directed Acyclic Graph (DAG)** of steps to execute the transformation
   - When the functions are used, they are passed from the **client** to the **master**
      - They are then applied across the partition of the RDD
- **DataFrames** are the primary abstracting in ***Spark SQL***
- **Spark SQL**: applies a schema to spark's RDDs
- **Turning a DataFrame into a local object**:
   - Some actions remain the same
   - Some new actions give the possibility to describe and show content in a more presentable format
   - when used/executed in IPython or in a notebook, they **launch the processing of the DAG**
      - This is where Spark ***stops being lazy***
- **Transformations on DataFrames**:
   - Still **lazy**: Spark does not apply transformation right away
      - Instead it builds on the **DAG**
   - Transform a DataFrame into another
      - DataFrames are **immutable**
   - Can be **wide** or **narrow**
   - work like RDDs because DataFrames are just RDDs with a schema applied
- ```udf()``` or ***user defined function*** turns a normal python function into something Spark can *parallelize* across its distributed data
   - Takes two arguments:
      1. A function
      2. The data type the function will return
- Spark has an ***SQL Interface*** for running queries in Python

## ***```WEEK FOUR: CAPSTONE ONE```***
### ***No Lectures!***

## :bomb:***```WEEK FIVE```***:bomb:

## K-Nearest Neighbors
- [**Link to Solutions**](https://github.com/GalvanizeDataScience/rfe1_solutions)
   - *Hit a snag in the assignment following along with the solutions*
### **Lecture Notes**
- **kNN** is a *supervised machine learning model* that can perform *Regression* and *Classification*
- y => Target variable (vector)  
- X => Features (matrix)
- **kNN Regression**: Set a number of neighbors *k* and find the ***mean*** of those values at a given point
- **kNN Classification**: Assigning the most common value in *k* at a given point
   1. *Training Algorithm*: Store all the data
      - Cheap training process, huge pro
   2. *Prediction Algorithm*: Predict the value of a new point, *x*
      - Calculate distance from *x* to all points in the dataset
         - Computationally expensive, drawback for *kNN*
      - Sort the points in your dataset by increasing distance from *x*
      - Get the mean of the target of the *k* closests points (aka nearest neighbors :smile:)
   - Euclidean distance is the most familiar metric
   - *sklearn* defaults to *Minkowski*, which is basically Euclidean
   - Manhattan can also be used (by setting p=1 in sklearn)
- ***Overfit*** occurs when the *k* is too low
- ***Underfit*** occurs when the *k* is too high
   - aka too smooth
- In general, choose *k* by setting it equal to the square root of *n*
- In order to check how successful a prediction was, the error *must* be measured
   - Popular methods include:
      - *Mean Squared Error (MSE)*
         - 𝑀𝑆𝐸:=1/𝑛∑𝑖=1𝑛(𝑦𝑖^−𝑦𝑖)2
      - *R-Squared*
         - More dependent on dataset than MSE
         - 𝑅2:=1−𝑆𝑆𝑟𝑒𝑠/𝑆𝑆𝑡𝑜𝑡
         - =1−∑𝑛𝑖=1(𝑦𝑖^−𝑦𝑖)2∑𝑛𝑖=1(𝑦𝑖−𝑦¯)2
         - Where *SSres* is the sum of squared residuals and *SStot* is the total sum of squares
         - Can be interpreted as the *fraction of the variance in the data that is explained by the model*
         - always between 0 and 1
         - ***Higher the better!***
- **Scaling Data**:
   - :bell:**Incredibly important concept according to Alex!**:bell:
   - Models that measure distance *must be scaled* so that everything is in the same format
   - *SKLearn* offers two main scalar functions:
      - Standard Scalar
      - MinMaxScalar
      - Using either of these is usually fine, at least for the purposes of the course
- **kNN (and other distance-based models) works generally well with lower dimensions dimensions**
   - With higher dimensional data, neighbors begin to become too far away to provide any use
   - ***"Curse of Dimensionality"***
- **Weighted Voting**: Using magnitude of the distance method to impact results
   - closer neighbors have stronger weight than those further away inside a given *k*
   - **Hard kNN** just returns classifications
   - **Soft kNN** returns probability and classes, using the number of neighbors of each class as the probability
- :bell:***Bias-Variance***:bell:
   - **Frequently comes up in interviews**
   - Concept in kNN is explained as overfitting and underfitting
## Bias-Variance/Cross Validation
- [**Link to Finished Assignment**](https://github.com/onewindspirit/cross-validation)
### **Lecture Notes**
- Basic concept is that the most optimal model is somewhere between over and underfitting
   - In kNN terms, the number of neighbors can't be too low or too high
- ***Error*** consists of **Bias**^2 + **Variance** + ***Irreducible Error***
- **High Bias**:
   - Model is *underfit*
   - Line is too rigid
   - Not enough features
   - Error tends towards one sit or the other in blocks
   - Error magnitude not randomly distributed
- **High Variance**:
   - Model is *overfit*
   - Line too flexible
   - Too many features
   - Errors tend to alternate positive/negative
   - Error magnitudes normally distributed
- **Optimal model has** ***minimum total error***
- **Cross Validation**
   - How it is determined which *model complexity* is "correct"
      - Complexity is defined per-model type
- **Cross-validation**:
   1. Attempts to quantify how well a model will predict on an unseen data set
   2. Tuning hyperparameters of models in order to get the best prediction
- **Cross-validation** step by step:
   1. Split your data (***after splitting out hold your set***) into training/validation sets
      - 70/30, 80/20, 90/10 are commonly used
         - determined by amount of data, high training values for large sets of data
   2. Use the training set to train several models of varying complexity
   3. Evaluate each model using the validation set
      - R^2, MSE, accuracy, etc.
   4. Keep the model that performs best over the **validation** set
- **K-Fold's Cross-validation**: After training get *k* estimates of validation error from the same complexity, so calculate the mean validation error from those *k* estimates
   - Gives a more robust, less variable estimate
   - A very common *k* is 5
- **Solutions to overfitting**
   - Get more data...
   - **Subset Selection**: Keep only a subset of your predictors
      - Try every model, every combination of *p* predictors
   - **Regularization**: restrict your model's parameter space
   - **Dimensionality Reduction**: Project the data into a lower dimensional space
## Predictive Linear Regression
- [**Link to Finished Assignment**](https://github.com/onewindspirit/predictive-linear-regression)
### **Lecture Notes**
- *A linear combination of a set of weights that align with particular inputs*
   - *Added all together for the y value*
- *Linear regression terminology flows throughout every other model*
   - *Foundational to the understanding of all other models*
- **Supervised learning**: Machine learning task of learning a function that maps an input to an output based on example input-output pairs
   - X,y -> predicting y based on the values in X
- **Unsupervised learning**: self-organized learning that helps find previously unknown patterns in data set without pre-existing labels
   - X -> understanding structure in X
   - *Is there a label or not for the thing you're looking for*
- **Parametric algorithm**:
   - fixed number of parameters
   - makes assumptions about structure
   - will work if the assumptions are correct
      - Fail silently if the parameters are incorrect
   - Examples:
      - linear regression
      - neural networks
      - stats distributions defined by a finite set of parameters
- **Non-parametric algorithm**:
   - Flexible number of parameters
      - Amount of parameters grow as the algorithm learns from more data
   - Makes fewer assumptions about data
   - Examples:
      - K-Nearest Neighbors
      - Decision Trees
- Given a data set of *n* statistical units, a linear regression model assumes that **the relationship between the dependent variable** ***y*** **and the p-vector of regressors** ***x*** **is linear**
   - ***When we increase x by 1, y should increase by 1 * Factor***
- **Simple linear regression**: Only one predictor leads to modeling the relationship
- **Multiple linear regression**: More than one predictor
   - Linear combinations
- The **coefficients** ***β*** are unknown
   - *Mathematical optimization*
   - In machine learning is typically discussed as making loss/cost functions
      - Goal of optimization is in finding the parameters/coefficients/weights that minimize the cost/loss function
- **Residual Sum of Squares (RSS)**
   - ∑𝑖=1𝑛(𝑦𝑖^−𝑦𝑖)2
   - Cost function
   - *Mean squared average* is the average RSS
- Any given set of parameters/coefficients will give a RSS value
   - Goal is to find the set that give the minimum RSS
   - This represents the *minimum of the cost function*
- Simple cost function: take the derivative, set it equal to 0, solve for coefficients
- Complex cost function: resort to numerical optimization like ***gradient descent***
- Assessing the accuracy/usefulness of your model:   
   - *How do the coefficients perform on your data?*
   - RSS is ok, but depends on *n*
   - MSE is better, but in squared units
   - Root Mean Square Error - in units of response
   - R^2 ranges from 0-1, but does not penalize for model complexity
   - *Adjusted* R^2 penalizes for model complexity according to number of predictors, *p*
- :rotating_light:***We should know at least our ROOT MSE and R^2 for every model we generate!***:rotating_light:
- Representing non-numerical data in regression models:
   - **Dummy variable**: encode presence of categorical features with a 1
      - Different 1's in columns for different categorical features
- Perform linear regression on *engineered features* to model linear regression on non-linear relationships
## Inferential Linear Regression
- [**Link to Finished Assignment**](https://github.com/onewindspirit/inferential-regression)
### **Lecture Notes**
- Predictive Linear Regression: Accurately predict a target
   - We care that it predicts well on unseen data
   - We don't care that:
      - Some features may be partially collinear
         - Cannot rely on our parameter estimates to tell us about the effect on the signal
      - Fundamental assumptions of inferential linear regression may be violated
- **Inferential Linear Regression**: Learn something accurate about the process of the data (*infer* coefficients)
   - models picked based on:
      - Trying different features
      - Feature engineering
      - Checking residuals
   - all this to see if we are **violating some assumptions of linear regression**
   - We care that parameter estimates are accurate and valid
   - We don't really care that it predicts well
      - but in theory it should
- **Inferential Linear Regression** is *only part of* understanding causality
   - We cannot deduce causality with ILR
   - We can only deduce *correlation* or *association*
   - ***Correlation does not imply causation***
- Linear Regression is beneficial because of the interpretability of its coefficients
   - If assumptions of inferential linear regression are being violated, parameter estimates are unreliable for providing interpretability
- **Assumptions of Inferential Linear Regression**:
   1. **Linearity**: Relationship between *X* and *y* can be modeled linearly
      - *Use pairwise scatterplots to visualize this*
      - If the data isn't linear:
         - **Transform the data** with log() or exponentiation of either the dependent or independent variable
         - **Try a nonlinear regression** (polynomial)
   2. **Independence**: The residuals should be independent from one another
      - Use a formal test for this *Durbin-Watson Test*
      - Look at a scatter plot of the *residuals*
   3. **Normality**: The residuals are normally distributed
      - Visualize using a *QQ plot* which measures divergence of the *residuals* from a normal distribution
         - Plotted data fits constant line in QQ Plot
      - Formal tests (Shapiro-Wilk, Kolmogorov-Smironov, Jarque-Bera or D'Agostino-Pearson)
         - Sensitive to large sample sizes
   4. **Homoscedasticity**: The variance of the residuals is constant
      - Visualize alongside *residual* plot
      - distributed evenly above and below across x
- :rotating_light:**Always look at your residuals!**:rotating_light:
   5. **No multicollinearity**: The independent variables are not highly correlated with each other
      - Two or more independent variables can *not* be highly correlated
      - Hard to detect in certain cases
      - *When looking at multicollinearity use* ***Variance Inflation Factor (VIF)***
- **VIF**: 1/1-(R^2)
   - Runs *Ordinary Least Squared Error* for each independent variable as a function of all the other predictors
      - K times for k predictors
   - VIF starts at 1
   - has no upper limit
   - VIF exceeding 5 or 10 indicates high multicollinearity between this independent variable and the others
   - VIF is an iterative process
      - May be multiple independent variables with a high VIF score
         - use best judgment in selecting variables for removal before re-calculating VIF
            - Generally remove the largest one
- *Multicollinearity does not reduce the predictive power or reliability of the model as a whole, at least within the sample data set, it only affects calculations regarding individual independent variables, which is why it isn’t necessarily always deemed as an assumption of inferential linear regression*
- Secondary Assumption: ***there are no influential outliers***
   - [***Anscombe's Quartet***](https://en.wikipedia.org/wiki/Anscombe%27s_quartet)
- If the assumptions aren’t met, it reduces the reliability of the model itself and the results may not be valid or meaningful
   - Often it is hard to tell if your assumptions are not met from the results
- **Confounding variable**:
   1. Correlated with the dependent variable
   2. Correlated with the independent variable
   3. Does not lie on the path from the independent variable to the dependent variable
   - Confounding variables may be the reason for association between independent and dependent variable
   - May be the reason there is no visible association between independent and dependent variable
- **Independent variables in regression**:
   1. **Continuous Variables**:
      - Holding all else constant, on average, a 1 unit increase in *independent variable* increases or decreases the *dependent variable* by the *estimated coefficient of the independent variable*
      - must decide whether to keep it:
         - good for interpretation
         - may not hold true with other assumptions
      - or *Standardize/Transform* it:
         - Harder for interpretation
         - Helps uphold assumptions
         - Allows for comparisons of estimated coefficients on the dependent variable
            - **not** importance of the variable itself
   2. Categorical Variables:
      - Binary encoding: 1's and 0's for boolean values
      - must create **dummy variables** in a process called ***one-hot encoding***
- ***Interpretation Sanity Check***
   1. Do the coefficients point in the correct direction?
      - Does the coefficient position sign make sense for that specific independent and dependent variable?
   2. Check the magnitude if you have a rough idea of what it should be
   3. Are the p-values significant?
      - Tells us whether the estimated coefficient differs from 0 or not
         - if so, it might be worth interpreting
- **F statistic hypothesis is if every coefficient is equal to 0**
- **What is the coefficient under the null hypothesis for the Z-Test? 0**

## Algorithmic Complexity
- [**Link to Finished Assignment**](https://github.com/onewindspirit/algorithmic-complexity)
### **Lecture Notes**
- **Computational Complexity and Big-O Notation**
   - Goal of computer programming is writing code that runs as quickly as possible
      - Not the most important goal
   - First step is understand how to measure the speed at which a particular program is running
      - ***Big-O Notation***
- **Big-O Notation**
   - Never so simple that we can just talk about how long a program takes to run
   - mathematical notation to describe **limiting behavior** of a function when the argument tends towards a given value or infinity
   - Constant factors are not of import
   - Interest in Big-O lies in **asymptotic behavior**
      - How a system behaves as the program's input length (*n*) trends towards infinity
   - Expressed as *"Big O* ***n*** *-squared"*
   - >*"If you let me multiply by some factor, I can always find some* ***n*** *at some point that will make this number more important than any of these other numbers"*
   - > *"As* ***n*** grows, ***n^2*** *will catch up"*
- ```%%timeit``` is used to measure the time it takes to run a cell *excluding* the first line
   - Does not work perfectly but is easy to use
- *For Loops* are usually *Big O of* ***n***
   - Increases exponentially with the amount of nested *For loops*
      - 2 for loops = *Big O of* ***n***^2, etc.
- In order to figure out the overall computational complexity of our code, we need to know the complexity for built-in functions.
   - ```sets```: are hashable
- **Best Case**: The element we're looking for is the first element
   - Does not happen very often
- **Average case**: Look at half the elements before we find our desired element
- **Worst case**: even though it is rare, we need to handle it without catastrophic failure
   - Might be common in real-world situations rather than random data
- **Sorted Lists**: operations can usually be performed faster if the data is organized
   - ```Bisect``` a list from the middle out to find the average case
#### :rotating_light:After the lecture, I like, don't understand what this is doing at a fundamental level:rotating_light:

## Regularized Regression
- [**Link to Solutions**](https://github.com/GalvanizeDataScience/rfe1_solutions)
   - *Solutions don't make no sense to me*
### **Lecture Notes**
- **Basic regular regression:**
   - The world is built on linear relationships
   - we can model the relationship between *features* and a *target*
   - We can make linear regression non-linear by inserting extra *interaction* features
   - We can *underfit* or *overfit* our model by sampling too little or too much
      - Add more variables to solve underfitting
      - *Regularize* the dataset in order to solve overfitting
         - Change the *cost function*
- **Curse of Dimensionality**:
   - In *higher dimensions* data is (usually) sparse
      - Linear regression has *high variance* (overfitting)
- **Ridge (L2) Regression**:
   - Linear regression with a regularization parameter added in that prefers *beta parameters* that are small
      - *Lambda* as a factor that states *how much* smaller betas are preferred
         - Just another hyperparameter to tune
   - If a feature's coefficient value (beta) becomes zero then **that features becomes irrelevant**
   - The intercept (Beta0) is not penalized
   - ***Increasing lambda increases the models bias and decreases its variance***
   - ***Only works on normalized data!***
   - Forces parameters to be small
   - Computationally easier than lasso because it is differentiable
- **LASSO (L1) Regression**:
   -  Lasso tends to set coefficients exactly equal to zero
      - *automatic feature selection* mechanism
      - leads to "sparse" models
      - Serves a similar purpose to stepwise features selection
- ***L2=taking the squares, L1=taking the absolute value***
- ***Make chart showing the two different types of regularization for case study/capstone 2***
   - ***Shows the effects of lamba in both versions of regression***
- Usually you should run both regressions and compare
   - ***Cross validation*** is how to determine which is best
- All are straightforward to call in sklearn:
   - ```sklearn.linear_model.LinearRegression(...)```
   - ```sklearn.linear_model.Ridge(alpha=my_alpha, …)```
   - ```sklearn.linear_model.Lasso(alpha=my_alpha, …)```
   - ```sklearn.linear_model.ElasticNet(alpha=my_alpha, l1_ratio = !!!!, …)```
   - **(In sklearn alpha = lambda)**
- ***You should probably use some amount of regularization in some form for any linear regression***

- ***After today, you'll have everything you need to start as a data scientist***

## Logistic Regression
- [**Link to Finished Assignment**](https://github.com/onewindspirit/logistic-regression)
### **Lecture Notes**
- **Logistic regression**: Supervised-learning parametric classification model used in some production environments
   - First thing you try when building a classifier
   - Used in some production environment
   - Advantages:
      - Fast (training and prediction)
      - Simple (few hyperparameters)
      - Interpretable
      - Provides probability
   - Disadvantages:
      - Requires feature engineering to capture non-linear relationships
      - Does not work for *p > n*
         - When you have more features than entries in the dataset
            - *More columns than rows*
- **Maximum Likelihood Estimation**: Technique to use in order to find out the *mu* and *sigma* of the data
   - Taking likelihoods of all the estimations we have
      - Shaping the distribution to find the highest value
   - In this case, MLE gives us *ordinary least squares*
- Logistic regression for sigmoid functions only work for 0s or 1s
   - aka bernoulli distributions
   - Logistic functions
- **Mathematical Odds**: The ratio of the probability of the positive to the negative case:
   - ```𝑂𝑅=𝑃(𝑦=1)/1−𝑃(𝑦=1)```
- **Logarithmic Odds**: Just log (base e) of the Mathematical Odds
   - Logistic function takes the *log odds* of something and returns the *probability*
      - ```𝑙𝑜𝑔(𝑂𝑅)=𝐗𝛽```
- **Decision Boundary**: Extends to a lot of models presented in future lectures
   - Consists of a **threshold**
      - The place at which something is put into one class or another
   - Decision boundary is the surface in the feature space at which the *probability is equal to the threshold*
      - Basically, drawing a line and seeing what residuals fall above and below the line
   - Formula for two features, *X1* and *X2*:
      - ```𝛽0+𝛽1𝑋1+𝛽2𝑋2=0```
- **Finding the beta coefficients:**
   - MLE can be used to construct a likelihood function:
      - ```𝑦∼𝐵𝑒𝑟𝑛𝑜𝑢𝑙𝑙𝑖(ℎ(𝑥⃗ ⋅𝛽⃗ ))```
   - For a Bernoulli trial with success probability *p*, the likelihood is:
      - ```𝑃(𝑦|𝑝)=𝑝𝑦(1−𝑝)1−𝑦```
   - In this case:
      - ```𝐿(𝛽⃗ )=∏𝑖ℎ(𝛽⃗ ⋅𝑥⃗ 𝑖)𝑦𝑖(1−ℎ(𝛽⃗ ⋅𝑥⃗ 𝑖))1−𝑦𝑖```
   - Setting the derivative of the likelihood to zero results in a set of nonlinear equations for 𝛽𝑗 that has **no analytical solution**
   - **Log likelihood is the loss function most commonly used here**
      - ```log𝐿(𝛽⃗ )=∑𝑖(𝑦𝑖log(ℎ(𝛽⃗ ⋅𝑥⃗ 𝑖))+(1−𝑦𝑖)log(1−ℎ(𝛽⃗ ⋅𝑥⃗ 𝑖)))```
- [**Log loss**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html) is simply the negative log likelihood defined above
   - How to score a *soft classifier*
- ### :rotating_light: ***Review loss functions in general*** :rotating_light:
- Unbalanced = ~<10%
- ***ceteris paribus***: a unit change in a feature value, results in a multiplication of the odds by a constant factor of 𝑒𝛽𝑖 where 𝛽𝑖 is the coefficient for that factor.
## Decision Rules
- [**Link to Finished Assignment**](https://github.com/onewindspirit/decision-rules)
### **Lecture Notes**
- **Confusion Matrix**: Gives the count of instances based on the actual and predicted values of the target
   - Used frequently
   - *True* and *False* refer to whether or not you are correct
   - *Positive* and *negative* refer to the **predicted** result
   - A *Type-1 error* is a false positive
   - Accuracy = ```𝑇𝑃+𝑇𝑁/𝑇𝑃+𝑇𝑁+𝐹𝑃+𝐹𝑁```
   - Sensitivity = Recall = TPR = 𝑇𝑃/𝑇𝑃+𝐹𝑁
      - True positive over all things that are positive
   - FPR ```𝐹𝑃/𝑇𝑁+𝐹𝑃```
      - False positive over all things that are negative
   - Specificity = ```𝑇𝑁/𝑇𝑁+𝐹𝑃```
   - Precision = PPV =```𝑇𝑃/𝑇𝑃+𝐹𝑃```
      - *Of all the apples you identified as poisoned, how many were actually poisoned?*
   - NPV =𝑇𝑁/𝑇𝑁+𝐹𝑁
   - Accuracy encompasses all of this
   - **F-Score**: 𝐹𝛽 evaluates a test assuming that recall is 𝛽 times as important as precision; it's a *weighted harmonic mean* of the two.
      - **Harmonic Mean**: instead of adding values together, you multiply them
   - **Receiver Operator Characteristic (ROC) Curves**: shows the *TPR (sensitivity)* vs. the *FPR (False Positive Rate or 1-Specificity)* for various thresholds
      - Alternative is the *Precision-recall curve*
         - more appropriate for looking in the positive class
      - Models can be compared by the ***Area Under the Curve (AUC)*** of either graph
- **Profit Curves and Imbalanced Classes**
   - Classification datasets can be *Imbalanced*
      - many observations of one class, fewer of another
   - The *cost* of a false positive is often different from the cost of a false negative
      - Need to consider external costs
   - Accuracy-driven models will over-predict the majority class
- **Solutions**:
   - Practical steps: help fit the model better
      - STratifying train_test_split
      - Change minority and majority weighting of training data
   - Cost-sensitive learning: use outside costs and benefits to set the prob. thresh
      - Thresholding (profit curves)
         - Quantify relative costs of TP,FP,TN,FN
         - Construct a *confusion matrix* for each probability threshold
         - Pick the threshold that gives highest profit
         - *signs of quantities in the cost-benefit matrix must be consistent*
         - *an easy mistake is to double-count by putting a benefit in one cell and a negative cost for the same thing in another cell (or vice versa*
   - Sampling: reduct imbalance with more or less data
      - Oversampling
         - Replicate observations from the minority class to balance training sample
         - Does not discard information **(PRO)**
         - Very likely to overfit **(CON)**
      - Undersampling
         - Randomly discards majority class observations to balance the training sample
         - Reduces runtime on large datasets **(PRO)**
         - Discards potentially important observations **(CON)**
      - **SMOTE**: Synthetic Minority Oversampling TEchnique
         - Generates new observations from minority class
         - For each minority class observation and for each feature:
            - Randomly generates between it and one of its *k-nearest neighbors*
         - Essentially, it creates new synthetic datapoints near the real datapoints
   - Neither cost sensitivity or sampling is strictly superior
      - Oversampling tends to work better than undersampling on small datasets
      - Some algorithms do not have an obvious cost-sensitive adaptation, meaning they require sampling