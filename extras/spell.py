import jamspell

corrector = jamspell.TSpellCorrector()
corrector.LoadLangModel('en.bin')

corrector.FixFragment('I am the begt spell cherken!')
# u'I am the best spell checker!'

corrector.GetCandidates(['i', 'am', 'the', 'begt', 'spell', 'cherken'], 3)
# (u'best', u'beat', u'belt', u'bet', u'bent', ... )

corrector.GetCandidates(['i', 'am', 'the', 'begt', 'spell', 'cherken'], 5)
# (u'checker', u'chicken', u'checked', u'wherein', u'coherent', ...)
