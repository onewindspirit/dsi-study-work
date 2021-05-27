# DSI Notes and Study Guide
Started this midway through week 2 but I'm sure it's gonna be helpful if I keep up with it.
 
## Numpy Notes:
Here are my notes while working through the Numpy assignment for the second time. Will update with review from Pre-Course later down the line.
### **Basics**
- `import numpy as np`
- `np.array()` is like the fundamental building block
 
### **Numpy Arrays**
**Numpy Arrays** appear listlike and may actions you can take on lists work for arrays, but arrays differ in that:
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
 
### **Broadcasting**
Operating on an array with another thing:
- Another array
- A single value (called scalar)
- A list (numpy converts this into an array automatically)
- Iterating with comparison objects (>,<,==,etc.) is **very useful** when combined with Boolean Indexing
- Have to use **operator symbols** (&,| rather than and, or)
- You can broadcast an array to another array but they must have complimentary shapes
 
### **Boolean Indexing**
This is when stuff starts to click. You can use an array of boolean values to select what elements of another array you return. Moreover, you can combine boolean indexing with broadcasting (especially in the cases where you're using comparison operations, mentioned above) to get to some cool places. This is really where the true nature and utility of numpy arrays seems to come to fruition; even though it's still not the first place my mind goes to when given a list of directions, it seems like I need to develop an instinct for when I need to  use these three concepts in tandem.
 
:bell:**I need to get the hang of nesting indexing and indexing in general**:bell:
 
- An example of using Boolean Indexing combined with Broadcasting:
   - Return only the rows where the value in the first column is bigger than five:
   ```
   array[array[:, 0] > 5, :]
   ```
 
### **Creating Arrays From Scratch**
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
 
### **Numpy.array Methods**
- Most conventional operations and methods can be called on arrays, though behavior is obviously more complex than it might seem
   - declaring an axis in your method offers slicing/more control
   - `keepdims=True` can be declared in order to keep the shape
 
 
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
- **When you decide the length of an experiment, you have to continue the experiment for the decide length**
- memorizing the math for these tests isn't necessary, but getting familiar with it will help when deciding what test to use in case studies and life in general
- **Mann-Whitney** is useful because it makes no assumptions about distribution
- **T Test** is expected when normality is expected
- Review StatsWeek youtube videos on this subject (in course outline)
   - Null Hypothesis, hypothesis testing and p-value videos specifically

