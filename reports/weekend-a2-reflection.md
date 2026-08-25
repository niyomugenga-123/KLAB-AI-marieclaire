# Reflection — Assignment 2

## Which transform took the longest to get right, and why?

The `Year` cleaning step was the hardest part, even though it isn't technically one of
the Task 3 transforms. `Year` had values like `20173` and `2013500` that looked like
random corruption at first. Figuring out the fix took a few steps: checking the
digit-length distribution of `Year`, noticing the corrupted rows lined up with
truncated `Name` values, and only then testing whether truncating to 4 digits actually
recovered plausible years (it did, for every corrupted row). Of the two Task 3
transforms, the groupby + merge was straightforward; the pivot_table took a bit more
trial and error to get the bin edges and labels right so the age and mileage bands
lined up cleanly.

## What would you do differently with another dataset this weekend?

I'd check the digit-length / value-range of every numeric column right after loading,
before doing anything else — that would have caught the `Year` corruption in Task 1
instead of during cleaning. I'd also decide the cleaning strategy for duplicates and
odd values earlier, since dropping ~57% of rows as exact duplicates changed the shape
of everything downstream.