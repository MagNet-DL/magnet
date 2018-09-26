`arghandle` is a small tool used to separate handling of arguments in functions from function logic.

Suppose you have a module `module.py` which defines an `add()` method for `int` or `float` numbers.

In addition, `int` is converted to `float`

Originally, you would write something as follows.

```python
# module.py

def add(x, y):
    for arg in (x, y):
        if not isinstance(arg, (int, float)):
            raise TypeError(f'Argument needs to be int or float.')
            
    if isinstance(x, int): x = float(x)
    if isinstance(y, int): y = float(y)
        
    return x + y
```

This pollutes the function because the main logic is obscured.

`arghandle` allows you to do the following:

* Create a separate module with the desired function in a folder called `__arghandle__`.
  Define your handling in that module.

  ```python
  # __arghandle__/module.py
  import arghandle
  
  from arghandle.handlers import typecheck
  def add(x, y):
      typecheck(x=x, y=y, include=(int, float))
              
      if isinstance(x, int): x = float(x)
      if isinstance(y, int): y = float(y)
          
      return arghandle.args()
  ```

* Now, your main function can be written as follows with just the function logic.

  ```python
  # module.py
  from arghandle import arghandle
  
  @arghandle
  def add(x, y):
      return x + y
  ```