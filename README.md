# Coding




sum of two numbers
class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        res = 0
        cur = 0
        for n in nums:
            if n:
                cur += 1 
                if cur > res:
                    res = cur
            else:
                cur = 0
        return res

     Palindrome Linked List
     class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        l=[]
        temp=head
        while(temp):
            l.append(temp.val)
            temp=temp.next
        low=0
        high=len(l)-1
        while(low<=high):
            if(l[low]!=l[high]):
                return False
            low+=1
            high-=1
        return True

    7 Reverse a String:

    class Solution:
    def reverse(self, x: int) -> int:
        sign = -1 if x < 0 else 1
        x *= sign
        reversed_x = 0
        while x:
            reversed_x = reversed_x * 10 + x % 10
            x //= 10
        reversed_x *= sign
        return 0 if reversed_x < -2**31 or reversed_x > 2**31 - 1 else reversed_x

        Leet Code 1838:
    class Solution:
    def maxFrequency(self, nums: list[int], k: int) -> int:
        nums.sort()
        left = 0
        total = 0
        result = 1

        for right in range(len(nums)):
            total += nums[right]

            # Cost to make all elements in window equal to nums[right]
            while nums[right] * (right - left + 1) > total + k:
                total -= nums[left]
                left += 1

            result = max(result, right - left + 1)

        return result
leet code (189):

class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        n = len(nums)
        k %= n

        def reverse(l, r):
            while l < r:
                nums[l], nums[r] = nums[r], nums[l]
                l += 1
                r -= 1

        reverse(0, n - 1)
        reverse(0, k - 1)
        reverse(k, n - 1)

Leet code(136):

class Solution:
    def singleNumber(self, nums):
        result = 0
        for num in nums:
            result ^= num
        return result

Leetcode (1):

class Solution:
    def twoSum(self, nums, target):
        hashmap = {}
        for i, num in enumerate(nums):
            if target - num in hashmap:
                return [hashmap[target - num], i]
            hashmap[num] = i
leetcode (169):

class Solution:
    def majorityElement(self, nums):
        candidate = None
        count = 0

        for num in nums:
            if count == 0:
                candidate = num
            count += 1 if num == candidate else -1

        return candidate

leetcode (53):
from typing import List

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        current_sum = nums[0]
        max_sum = nums[0]
        
        for num in nums[1:]:
            current_sum = max(num, current_sum + num)
            max_sum = max(max_sum, current_sum)
        
        return max_sum

leet code(121):

from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_price = float('inf')
        max_profit = 0
        
        for price in prices:
            if price < min_price:
                min_price = price
            else:
                max_profit = max(max_profit, price - min_price)
                
        return max_profit
        
leetcode(2149):

from typing import List

class Solution:
    def rearrangeArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n
        
        pos_index = 0
        neg_index = 1
        
        for num in nums:
            if num > 0:
                result[pos_index] = num
                pos_index += 2
            else:
                result[neg_index] = num
                neg_index += 2
        
        return result

leetcode (31):

class Solution:
    def nextPermutation(self, nums: list[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        
        # Step 1: Find first decreasing element from right
        i = n - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        
        # Step 2: If found, find element just larger than nums[i]
        if i >= 0:
            j = n - 1
            while nums[j] <= nums[i]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        
        # Step 3: Reverse the suffix
        left, right = i + 1, n - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
leetcode (73):
from typing import List

class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        rows = len(matrix)
        cols = len(matrix[0])
        
        first_col_zero = False
        
        # Step 1: Mark rows and columns that need to be zeroed
        for i in range(rows):
            if matrix[i][0] == 0:
                first_col_zero = True
            for j in range(1, cols):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
        
        # Step 2: Use markers to set zeroes (excluding first row and column)
        for i in range(1, rows):
            for j in range(1, cols):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        
        # Step 3: Handle first row
        if matrix[0][0] == 0:
            for j in range(cols):
                matrix[0][j] = 0
        
        # Step 4: Handle first column
        if first_col_zero:
            for i in range(rows):
                matrix[i][0] = 0

leet code(38):
class Solution:
    def countAndSay(self, n: int) -> str:
        result = "1"
        
        for _ in range(n - 1):
            current = ""
            count = 1
            
            for i in range(1, len(result)):
                if result[i] == result[i - 1]:
                    count += 1
                else:
                    current += str(count) + result[i - 1]
                    count = 1
            
            # Add the last group
            current += str(count) + result[-1]
            result = current
        
        return result
leetcode (48):
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)
        
        # Step 1: Transpose the matrix
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        
        # Step 2: Reverse each row
        for row in matrix:
            row.reverse()
leetcode (58):
from typing import List

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix or not matrix[0]:
            return []
        
        result = []
        top, bottom = 0, len(matrix) - 1
        left, right = 0, len(matrix[0]) - 1
        
        while left <= right and top <= bottom:
            
            # Traverse from left to right
            for col in range(left, right + 1):
                result.append(matrix[top][col])
            top += 1
            
            # Traverse from top to bottom
            for row in range(top, bottom + 1):
                result.append(matrix[row][right])
            right -= 1
            
            # Traverse from right to left
            if top <= bottom:
                for col in range(right, left - 1, -1):
                    result.append(matrix[bottom][col])
                bottom -= 1
            
            # Traverse from bottom to top
            if left <= right:
                for row in range(bottom, top - 1, -1):
                    result.append(matrix[row][left])
                left += 1
        
        return result

leetcode (560):

from typing import List

class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        count = 0
        current_sum = 0
        prefix_sum = {0: 1}  # base case

        for num in nums:
            current_sum += num
            
            if current_sum - k in prefix_sum:
                count += prefix_sum[current_sum - k]
            
            prefix_sum[current_sum] = prefix_sum.get(current_sum, 0) + 1

        return count








