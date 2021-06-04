# -*- coding: utf-8 -*-
import os
import re

from pyrouge import Rouge155
import tempfile

class Rouge(object):
    def __init__(self, settings):
        self.settings = settings

    def eval(self, hypList, refList):
        if not os.path.isdir("./temp"):
            os.system("mkdir ./temp")
        tempfile.tempdir = "./temp"

        number = len(hypList)
        n_ref = len(refList) // number
        # Generate Files
        os.system("rm " + self.settings['rouge_hyp_dir'] + "*")
        os.system("rm " + self.settings['rouge_ref_dir'] + "*")

        for Index in range(0, number):
            hypName = self.settings['rouge_hyp_dir'] + 'hyp.' + str(Index).zfill(6) + '.txt'
            f1 = open(hypName, 'w')
            # f1.write(RougeTrick(hypList[Index]))
            f1.write(hypList[Index])
            f1.close()

            for i in range(n_ref):
                refName = self.settings['rouge_ref_dir'] + 'ref.' + chr(ord('A') + i) + '.' + str(Index).zfill(
                    6) + '.txt'
                f2 = open(refName, 'w')
                # f2.write(RougeTrick(refList[Index * n_ref + i]))
                f2.write(refList[Index * n_ref + i])
                f2.close()

        if number == 500:
            # R = Rouge155('./ROUGE-RELEASE-1.5.5', '-e ./ROUGE-RELEASE-1.5.5/data -n 4 -m -w 1.2 -c 95 -r 1000 -b 75 -a')
            R = Rouge155('./ROUGE-RELEASE-1.5.5')
        else:
            R = Rouge155('./ROUGE-RELEASE-1.5.5')
        R.system_dir = self.settings['rouge_hyp_dir']
        R.model_dir = self.settings['rouge_ref_dir']
        R.system_filename_pattern = 'hyp.(\d+).txt'
        R.model_filename_pattern = 'ref.[A-Z].#ID#.txt'

        output = R.convert_and_evaluate()
        output_dict = R.output_to_dict(output)
        os.system("rm -rf ./temp/*")
        return output_dict
