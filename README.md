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
