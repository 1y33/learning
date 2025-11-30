What is Redis?
Redis is a popular in memory cache system for storing data in good data structures ( most of them for the O(1) retrieval)

An example lets say if we have a system and we stumble upon on this stuff:
- we have a lot of reused values
- the system is failng unpredictably due to outside problems
- We want to maintain user data even after leaving the aplication after a crash lets say ( so we knwo that the user will rejoin in 10-20 seconds)

Then redis is the solution to it