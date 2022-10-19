import unittest
import re
from valla.dsets.blogs import get_posts_from_file


class TestBlogExtract(unittest.TestCase):

    def setUp(self) -> None:
        file = '/Users/jtyo/datasets/blogs/987163.female.23.Education.Leo.xml'
        self.extracted = [x for x, _ in get_posts_from_file(file)]

        self.truth = [
            "Arun:  I'm not deviant! Mell:  You're Canadian! Arun:  It's morally sound world domination!  (yes, these are two of the main influences in my life... now you wonder why I NEED a weblog...)",
            "Today on piano:    Phantom of the Opera   Today on guitar:  everything I know how to play...  Today on TV:  Debate about Iraq resolutions.  Lots of words were said, few were actually communicated.",
            "I went to the LACMA today with some ex-C2 buddies (that's ex-C2, not ex-buddy).  I had a really good time.  Ricky got to look at his piece of Southeast Asian Art, Olga and I got to see our century of fashion, and Lucas got to see George Washington even though the rest of us weren't too keen on that.  THEN... and THEN... we saw the Donald Blumberg exhibit!!  It was amazing!!  I had never heard of this guy before, but his work was absolutely amazing.  The exhibit is mostly black and white photography (simple room, basic black frames with white mattes).  My favorites were the TV series and the Daily Series... If you get a chance before January 5th, hop on over to the LACMA ($5 for students).  Here's a little clip from the LACMA press release:     Described as a consummate technician and photographers photographer, Blumberg is an artist whose signature style rests not in his selections of subject matter or in how those subjects are portrayed, but in the manner in which his images address issues of the photographic... The importance of Blumbergs work lies in its reconciliation of the cognitive with the visual, the conceptual with the visceral, and the common with the exceptional. Donald Blumbergs art is an art of integration.",
            "I've been trying to figure out what to research for my final paper for my education policy class.  I always have the hardest time deciding on a project when the assignment is left so open-ended.  On first inclination, I started thinking about race issues.  This all links to affirmative action, the definition of racial subgroups in terms of evaluating public schools, the upcoming Racial Privacy Initiative that will be on the 2004 ballot.  Then I turned my focus to reading curriculum.  Think about how you learned how to read.  I can't remember exactly how I did.  California just implemented a phonics-based reading system in 1999.  LAUSD specifically uses what's called the \"Open Court\" reading system that is a rigorous 3-hr a day phonics-based curriculum.  Now that sounds like a good thing, because for decades people were leaving high school with a diploma but without basic reading skills, so this was an issue they knew they had to attack from the beginning.  But we're looking at a rigid curriculum that has improved reading scores, but that must take away from other things important to an elementary education, say, science, social studies, history, music, art...     Then I realized that there was an underlying issue beneath all of this.  Quick-fix policy vs. long-term-change policy.  Everything that is implemented is quick-fix.  Kids can't read, so let's do something drastic that may take away from other education opportunities, but it's ok as long as their reading scores go up by the first year.  There are racial injustices, so let's reverse that by accounting for the hardships that minorities face in admissions processes.  All quick-fix.  In the long run, these systems don't hit the core of the problems.  But what's the alternative to quick-fix?  There really is no alternative.  We have government officials who are in office for such a short amount of time that they have to prove themselves in two years before election time rolls around again.  So if you were in office what would you want to do?  Implement some policy that will show significant results before it's time for people to vote for you again.  Then it looks like you actually did something.  And think about if you were the person in charge of all the money.  You have such a limited pool of resources to work with, how do you decide how much goes where?  You give a system a certain amount of money and that system has to prove itself worthy before it comes time to redelegate money, otherwise it risks losing that funding for the next term.  So again, what does it have to do?  Produce significant results in two years.  But things really don't work this way.  Nothing can be fixed in such a short amount of time.   Long-term change happens slowly... results may not show up for years... Now that I've complained about that, do I have a solution?  Nope.  Kinda sad though.     \"Because you can't change the world, but you can make a dent in it.\"    from Death to Smoochy",
            "\"everytime she sneezes I believe it's love.\"     from Anna Begins, Counting Crows"
        ]

    def test_extraction(self):

        print('manual check')
        print('Truth =================')
        for txt in self.truth:
            print(txt)
        print('Extracted =============')
        for txt in self.extracted:
            print(txt)

        for txt1, txt2 in zip(self.extracted, self.truth):
            print(txt1)
            print(txt2)
            print(re.sub('/^![a-z0-9]+$/i', '', txt1).strip() == re.sub('/^![a-z0-9]+$/i', '', txt2).strip())

        self.assertEqual(len(self.truth), len(self.extracted), 'different lengths O.o')
        # for txt in self.truth:
        #     print(txt)
        #
        #     self.assertIn(txt, self.extracted, f'{txt} not found in extracted')

        self.assertSetEqual(set(self.truth), set(self.extracted), f'sets not equal.')

    def tearDown(self) -> None:
        pass


if __name__ == '__main__':
    unittest.main()
