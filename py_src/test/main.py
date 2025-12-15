#!/usr/bin/env python3
import numpy as np
import fast_module  # This is the compiled C++ module

# 1. Instantiate the C++ class
N = 5  # Keeping it small for display, but this works for N = 10^7+
print("--- Python: Creating C++ Object ---")
processor = fast_module.DataProcessor(N)

# 2. Get the data as a NumPy array (Zero-Copy View)
# This array actually points to the memory inside the C++ object.
numpy_view = processor.get_data_view()

print(f"\n--- Python: Initial Data (NumPy View) ---")
print(numpy_view)

# 3. Modify data using the C++ method
print("\n--- Python: Calling C++ method 'add_scalar(10.0)' ---")
processor.add_scalar(10.0)

# Check the NumPy view again (it should update automatically)
print(f"Data in NumPy after C++ modification: {numpy_view}")

# 4. Modify data using NumPy (Python side)
print("\n--- Python: Modifying via NumPy (array += 5.0) ---")
numpy_view += 5.0

# Verify in C++? (If we called add_scalar again, it would see the new values)
processor.add_scalar(1.0)
print(f"Data after Python modification + C++ addition: {numpy_view}")

print("\n--- Python: End of script ---")
# The C++ destructor will be called here when 'processor' goes out of scope.
