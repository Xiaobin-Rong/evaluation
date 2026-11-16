import os
from glob import glob


def merge_results(netout_folder):
    results_list = sorted(glob(os.path.join(netout_folder, "*", "RESULTS.txt")))
    summary_file = os.path.join(netout_folder, "RESULTS.txt")
    with open(summary_file, 'w', encoding='utf-8') as outfile:
        for filename in results_list:
            with open(filename, 'r', encoding='utf-8') as infile:
                content = infile.read()
                outfile.write(content)
                outfile.write('\n')
    print(f'Overall results have been written in {summary_file}')
    

def main(args):
    netout_folder = args.inf_path
    assert os.path.exists(f'{netout_folder}/inf.scp'), "No inf scp files"
    assert os.path.exists(f'{netout_folder}/ref.scp'), "No ref scp files"

    if 'dnsmos' in args.metric:
        os.system(
            ('python ./evaluation_metrics/calculate_nonintrusive_dnsmos.py '
                f'--inf_scp {netout_folder}/inf.scp '
                f'--output_dir {netout_folder}/scoring_dnsmos '
                '--device cuda '
                '--job 1 '
                '--convert_to_torch True '
                '--primary_model ./lib/DNSMOS/sig_bak_ovr.onnx '
                '--p808_model ./lib/DNSMOS/model_v8.onnx'
            )
        )
        if args.display:
            os.system(f"cat {netout_folder}/scoring_dnsmos/RESULTS.txt")

    if 'utmos' in args.metric:
        os.system(
            ('python ./evaluation_metrics/calculate_nonintrusive_mos.py '
                f'--inf_scp {netout_folder}/inf.scp '
                f'--output_dir {netout_folder}/scoring_utmos '
                '--device cuda '
                '--job 1 '
            )
        )
        if args.display:
            os.system(f"cat {netout_folder}/scoring_utmos/RESULTS.txt")

    if 'nisqa' in args.metric:
        os.system(
            ('python ./evaluation_metrics/calculate_nonintrusive_nisqa.py '
                f'--inf_scp {netout_folder}/inf.scp '
                f'--output_dir {netout_folder}/scoring_nisqa '
                '--device cuda '
                '--job 1 '
                '--nisqa_model ./lib/NISQA/weights/nisqa.tar'
            )
        )
        if args.display:
            os.system(f"cat {netout_folder}/scoring_nisqa/RESULTS.txt")
            
    if 'plcmos' in args.metric:
        os.system(
            ('python ./evaluation_metrics/calculate_nonintrusive_plcmos.py '
                f'--inf_scp {netout_folder}/inf.scp '
                f'--output_dir {netout_folder}/scoring_plcmos '
                '--device cuda '
                '--job 1 '
            )
        )
        if args.display:
            os.system(f"cat {netout_folder}/scoring_plcmos/RESULTS.txt")
            
    if 'intrusive' in args.metric:
        os.system(
            ('python ./evaluation_metrics/calculate_intrusive_se_metrics_v3.py '
             f'--ref_scp {netout_folder}/ref.scp '
             f'--inf_scp {netout_folder}/inf.scp '
             f'--output_dir {netout_folder}/scoring_intrusive '
             '--nj 8 '
             '--chunksize 1000'
            )
        )
        if args.display:
            os.system(f"cat {netout_folder}/scoring_intrusive/RESULTS.txt")

    if 'phnsim' in args.metric:
        os.system(
            ('python ./evaluation_metrics/calculate_phoneme_similarity.py '
             f'--ref_scp {netout_folder}/ref.scp '
             f'--inf_scp {netout_folder}/inf.scp '
             f'--output_dir {netout_folder}/scoring_phnsim '
             '--device cuda '
             '--nsplits 1 '
            #  'job 1'
            )
        )
        if args.display:
            os.system(f"cat {netout_folder}/scoring_phnsim/RESULTS.txt")

    if 'spksim' in args.metric:
        os.system(
            ('python ./evaluation_metrics/calculate_speaker_similarity.py '
             f'--ref_scp {netout_folder}/ref.scp '
             f'--inf_scp {netout_folder}/inf.scp '
             f'--output_dir {netout_folder}/scoring_spksim '
             '--device cuda '
             '--nsplits 1 '
            #  'job 1'
            )
        )
        if args.display:
            os.system(f"cat {netout_folder}/scoring_spksim/RESULTS.txt")

    if 'spksim_wavlm' in args.metric:
        os.system(
            ('python ./evaluation_metrics/calculate_speaker_similarity_wavlm.py '
             f'--ref_scp {netout_folder}/ref.scp '
             f'--inf_scp {netout_folder}/inf.scp '
             f'--output_dir {netout_folder}/scoring_spksim_wavlm '
             '--device cuda '
             '--nsplits 1 '
            #  'job 1'
            )
        )
        if args.display:
            os.system(f"cat {netout_folder}/scoring_spksim_wavlm/RESULTS.txt")
            
    if 'spksim_wavlm_ft' in args.metric:
        os.system(
            ('python ./evaluation_metrics/calculate_speaker_similarity_unispeech_wavlm.py '
             f'--ref_scp {netout_folder}/ref.scp '
             f'--inf_scp {netout_folder}/inf.scp '
             f'--output_dir {netout_folder}/scoring_spksim_wavlm_ft '
             '--device cuda '
             '--nsplits 1 '
            #  'job 1'
            )
        )
        if args.display:
            os.system(f"cat {netout_folder}/scoring_spksim_wavlm_ft/RESULTS.txt")

    if 'bertscore' in args.metric:
        os.system(
            ('python ./evaluation_metrics/calculate_speechbert_score_v2.py '
             f'--ref_scp {netout_folder}/ref.scp '
             f'--inf_scp {netout_folder}/inf.scp '
             f'--output_dir {netout_folder}/scoring_bertscore '
             '--device cuda '
             '--nsplits 1 '
            #  'job 1'
            )
        )
        if args.display:
            os.system(f"cat {netout_folder}/scoring_bertscore/RESULTS.txt")
            
    if 'mbertscore' in args.metric:
        os.system(
            ('python ./evaluation_metrics/calculate_speechbert_score_v3.py '
             f'--ref_scp {netout_folder}/ref.scp '
             f'--inf_scp {netout_folder}/inf.scp '
             f'--output_dir {netout_folder}/scoring_mbertscore '
             '--device cuda '
             '--nsplits 1 '
            #  'job 1'
            )
        )
        if args.display:
            os.system(f"cat {netout_folder}/scoring_mbertscore/RESULTS.txt")

    if 'asr' in args.metric:
        os.system(
            ('python ./evaluation_metrics/infer_asr.py '
             f'--inf_scp {netout_folder}/inf.scp '
             f'--output_dir {netout_folder}/inferred_metadata '
             '--device cuda '
             '--nsplits 1 '
            #  'job 1'
            )
        )
        
    if 'wer' in args.metric:  # Eng only
        os.system(
            ('python ./evaluation_metrics/calculate_wer.py '
             f'--meta_tsv {args.text} '
             f'--inf_scp {netout_folder}/inf.scp '
             f'--output_dir {netout_folder}/scoring_wer '
             '--device cuda '
             '--nsplits 1 '
            #  'job 1'
            )
        )
        if args.display:
            os.system(f"cat {netout_folder}/scoring_wer/RESULTS.txt")

    if 'cer' in args.metric:  # Multi-lingual
        os.system(
            ('python ./evaluation_metrics/calculate_cer.py '
             f'--inf_scp {netout_folder}/inf.scp '
             f'--meta_tsv {args.text} '
             f'--utt2lang {args.lang} '
             f'--output_dir {netout_folder}/scoring_cer '
             '--device cuda '
             '--nsplits 1 '
            #  'job 1'
            )
        )
        if args.display:
            os.system(f"cat {netout_folder}/scoring_cer/RESULTS.txt")
    
    if 'wer_whisper' in args.metric:  # Eng only
        os.system(
            ('python ./evaluation_metrics/calculate_wer_whisper.py '
             f'--meta_tsv {args.text} '
             f'--inf_scp {netout_folder}/inf.scp '
             f'--output_dir {netout_folder}/scoring_wer_whisper '
             '--device cuda '
            )
        )
        if args.display:
            os.system(f"cat {netout_folder}/scoring_wer_whisper/RESULTS.txt")

    if 'cer_whisper' in args.metric:  # Multi-lingual
        os.system(
            ('python ./evaluation_metrics/calculate_cer_whisper.py '
             f'--inf_scp {netout_folder}/inf.scp '
             f'--meta_tsv {args.text} '
             f'--utt2lang {args.lang} '
             f'--output_dir {netout_folder}/scoring_cer_whisper '
             '--device cuda '
            )
        )
        if args.display:
            os.system(f"cat {netout_folder}/scoring_cer_whisper/RESULTS.txt")

    if 'scoreq' in args.metric:
        os.system(
            ('python ./evaluation_metrics/calculate_nonintrusive_scoreq.py '
            f'--inf_scp {netout_folder}/inf.scp '
            f'--output_dir {netout_folder}/scoring_scoreq '
            # '--device cuda '
            '--job 1 '
            )
        )
        if args.display:
            os.system(f"cat {netout_folder}/scoring_scoreq/RESULTS.txt")

    if 'emosim' in args.metric:
        os.system(
            ('python ./evaluation_metrics/calculate_emotion_similarity.py '
            f'--ref_scp {netout_folder}/ref.scp '
            f'--inf_scp {netout_folder}/inf.scp '
            f'--output_dir {netout_folder}/scoring_emosim '
            f'--device cuda '
            )
        )
        if args.display:
            os.system(f"cat {netout_folder}/scoring_emosim/RESULTS.txt")

    if 'lidacc' in args.metric:
        os.system(
            ('python ./evaluation_metrics/calculate_lid_accuracy.py '
            f'--meta_tsv {args.lang} '
            f'--inf_scp {netout_folder}/inf.scp '
            f'--output_dir {netout_folder}/scoring_lidacc '
            f'--device cuda '
            )
        )
        if args.display:
            os.system(f"cat {netout_folder}/scoring_lidacc/RESULTS.txt")


    if args.summary:
        merge_results(netout_folder)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--inf_path')
    parser.add_argument('-T', '--text', default="")
    parser.add_argument('-L', '--lang', default="")
    parser.add_argument(
        '-M',
        '--metric',
        nargs='+',
        default=['dnsmos', 'nisqa', 'utmos',
                 'intrusive',
                 'phnsim', 'spksim', 'bertscore'],
        help="Metric to be calculated"
    )
    parser.add_argument('-D', '--device', default='0')
    parser.add_argument('-S', '--summary', default=True)
    parser.add_argument('--display', default=True)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main(args)
