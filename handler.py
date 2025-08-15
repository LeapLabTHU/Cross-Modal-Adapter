from collections import OrderedDict
from pathlib import Path

import pandas as pd
import yaml


class ExpHandler:
    """
    en_wandb, debug, no_resume, output_dir
    """

    def __init__(self, args):
        self._save_dir = Path(args.output_dir)
        self._run_name = self._save_dir.name

        # self.csv_path = self._save_dir / f'{self._run_name}.csv'
        # self.cfg_path = self._save_dir / f'{self._run_name}.yaml'
        self.csv_path = self._save_dir / 'log.csv'
        self.cfg_path = self._save_dir / 'args.yaml'

        if args.en_wandb:
            import wandb
            wandb_kwargs = dict(project=self._save_dir.parents[1].name,
                                group=self._save_dir.parents[0].name,
                                name=self._run_name,
                                save_code=True,
                                resume='allow')

        args.commit = self._get_commit_hash()
        if not args.no_resume and self.cfg_path.exists():
            with open(self.cfg_path, 'r') as f:
                config = yaml.safe_load(f)
            if args.en_wandb:
                if 'wandb_id' not in config:
                    wandb_kwargs['id'] = wandb.util.generate_id()
                else:
                    wandb_kwargs['id'] = config['wandb_id']
            self.log_data = pd.read_csv(self.csv_path).to_dict('records') if self.csv_path.exists() else []
        else:  # new run
            if args.en_wandb:
                wandb_kwargs['id'] = wandb.util.generate_id()
            self._save_config(args)
            self.log_data = []

        if args.en_wandb:
            self.wandb_run = wandb.init(**wandb_kwargs)


    @staticmethod
    def _get_commit_hash():
        import git
        try:
            repo = git.Repo(search_parent_directories=True)
            commit = repo.head.object.hexsha
        except git.InvalidGitRepositoryError:
            commit = 'not_set'
        return commit

    @property
    def save_dir(self):
        return self._save_dir

    def _save_config(self, args):
        conf = vars(args)
        if hasattr(self, 'wandb_run'):
            conf['wandb_id'] = self.wandb_run.id

        print('=' * 40)
        for k, v in conf.items():
            print(f'{k}: {v}')
        print('=' * 40)

        with open(self.cfg_path, 'w') as f:
            yaml.safe_dump(conf, f, sort_keys=False)

    def write(self, eval_metrics=None, train_metrics=None, **kwargs):
        rowd = OrderedDict([(f'{k}', v) for k, v in kwargs.items()])
        if train_metrics:
            rowd.update([(f'train/' + k, v) for k, v in train_metrics.items()])
        if eval_metrics:
            rowd.update([(f'eval/' + k, v) for k, v in eval_metrics.items()])
        self.log_data.append(rowd)

        pd.DataFrame(self.log_data).to_csv(self.csv_path,
                                           index=False)
        # initial = not os.path.exists(self.csv_path)
        # with open(self.csv_path, mode='a') as cf:
        #     dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        #     if initial:
        #         dw.writeheader()
        #     dw.writerow(rowd)

        if hasattr(self, 'wandb_run'):
            self.wandb_run.log(rowd)

    def finish(self):
        (self._save_dir / 'finished').touch()
        if hasattr(self, 'wandb_run'):
            self.wandb_run.finish()

    @property
    def wandb_obj(self):
        if hasattr(self, 'wandb_run'):
            return self.wandb_run
