"""attach_priors（エネ付与先の帯実測注入）のテスト.

観測は cabt を通さず軽量フェイク（SimpleNamespace）で組む。
カード ID はメタから拾った実在 ID をダミー扱いで使う（Pokémon Elements を持ち込まない）。
"""

import json
import os
import random
import sys
from collections import defaultdict
from types import SimpleNamespace as NS

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

pytest.importorskip("cg.api", reason="cabt Engine (cg) が見つからない")

from agents import _choose_main, make_heuristic_agent  # noqa: E402
from cards import load_card_meta  # noqa: E402
from cg.api import AreaType, OptionType, SelectType  # noqa: E402
from mine_attach_policy import mine_episode  # noqa: E402
from submission import _load_attach_priors  # noqa: E402


@pytest.fixture(scope="module")
def meta():
    return load_card_meta()


@pytest.fixture(scope="module")
def basic_id(meta):
    return next(c for c in meta.card_type if meta.is_basic_pokemon(c))


def _pk(cid, n_energy=0, hp=100):
    return NS(id=cid, energyCards=[NS(id=0)] * n_energy, hp=hp)


def _attach_obs(cid, act_e=0, bench_e=(0, 1)):
    """MAIN で active/bench 両方へのエネ付与が選べる観測フェイクを組む.

    option[0]=active 付与・option[1..]=bench 付与（inPlayIndex がベンチ枠に対応）。
    """
    opts = [NS(type=OptionType.ATTACH, inPlayArea=AreaType.ACTIVE, inPlayIndex=0, index=0)]
    bench = []
    for b, n in enumerate(bench_e):
        opts.append(
            NS(type=OptionType.ATTACH, inPlayArea=AreaType.BENCH, inPlayIndex=b, index=0)
        )
        bench.append(_pk(cid, n))
    me = NS(active=[_pk(cid, act_e)], bench=bench)
    st = NS(yourIndex=0, players=[me, NS()], turn=1)
    sel = NS(type=SelectType.MAIN, option=opts, minCount=1, maxCount=1)
    return NS(current=st, select=sel)


def test_without_priors_prefers_active(meta, basic_id):
    """priors 未指定は従来挙動: アクティブ優先（挙動不変の保証）."""
    obs = _attach_obs(basic_id)
    got = _choose_main(obs, meta, random.Random(0), use_trainers=False)
    assert obs.select.option[got].inPlayArea == AreaType.ACTIVE


def test_priors_send_to_bench(meta, basic_id):
    """P(ベンチ)=1.0 なら必ずベンチへ、0.0 なら従来通りアクティブへ."""
    obs = _attach_obs(basic_id, act_e=0)
    got = _choose_main(
        obs, meta, random.Random(0), use_trainers=False, attach_priors={0: 1.0}
    )
    assert obs.select.option[got].inPlayArea == AreaType.BENCH
    got = _choose_main(
        obs, meta, random.Random(0), use_trainers=False, attach_priors={0: 0.0}
    )
    assert obs.select.option[got].inPlayArea == AreaType.ACTIVE


def test_priors_bucket_clamps_to_max_key(meta, basic_id):
    """アクティブのエネ枚数が採掘バケット上限を超えたら最大キーの確率を使う."""
    obs = _attach_obs(basic_id, act_e=5)  # 5枚 → キー {0,1} の最大=1 を参照
    got = _choose_main(
        obs, meta, random.Random(0), use_trainers=False,
        attach_priors={0: 0.0, 1: 1.0},
    )
    assert obs.select.option[got].inPlayArea == AreaType.BENCH


def test_bench_target_prefers_charged(meta, basic_id):
    """ベンチ付与先は「攻撃準備に近い子」＝装着エネが多い方を選ぶ（同一 id 同士）."""
    obs = _attach_obs(basic_id, act_e=2, bench_e=(0, 2))
    got = _choose_main(
        obs, meta, random.Random(0), use_trainers=False, attach_priors={0: 1.0}
    )
    opt = obs.select.option[got]
    assert opt.inPlayArea == AreaType.BENCH and opt.inPlayIndex == 1


def test_heuristic_agent_passes_attach_priors(meta, basic_id):
    """make_heuristic_agent 経由でも attach_priors が効く."""
    agent = make_heuristic_agent(meta, use_trainers=False, attach_priors={0: 1.0})
    obs = _attach_obs(basic_id)
    got = agent(obs, random.Random(0))
    assert obs.select.option[got[0]].inPlayArea == AreaType.BENCH


def test_load_attach_priors_reads_json(tmp_path):
    """_load_attach_priors: priors キー形式を int キーで読む。無ければ None."""
    p = tmp_path / "attach_priors.json"
    p.write_text(json.dumps({"priors": {"0": 0.49, "3": 0.77}}))
    assert _load_attach_priors(str(p)) == {0: 0.49, 3: 0.77}
    assert _load_attach_priors(str(tmp_path / "missing.json")) is None


def test_mine_episode_counts_choice_only_when_both_areas(tmp_path, basic_id):
    """採掘: active/bench 両方が合法だった MAIN 決定だけを数え、付与先を集計する."""
    def opt_d(area, ipi=0):
        return {"type": OptionType.ATTACH, "inPlayArea": area, "inPlayIndex": ipi,
                "index": 0}

    def step(status, opts, act_e, action=None):
        obs = {
            "select": {"type": SelectType.MAIN, "option": opts},
            "current": {
                "yourIndex": 0,
                "players": [
                    {"active": [{"id": basic_id,
                                 "energyCards": [{"id": 0}] * act_e}],
                     "bench": []},
                    {},
                ],
            },
        }
        return {"status": status, "observation": obs, "action": action}

    both = [opt_d(AreaType.ACTIVE), opt_d(AreaType.BENCH)]
    only_active = [opt_d(AreaType.ACTIVE)]
    ep = {
        "info": {"TeamNames": ["T", "U"]},
        "steps": [
            # 1手目: 両方あり・エネ2枚 → bench(=index1) を選択 ⇒ バケット2に bench+1
            [step("ACTIVE", both, 2), {"status": "INACTIVE"}],
            [step("INACTIVE", [], 0, action=[1]), {"status": "INACTIVE"}],
            # 2手目: active しか無い → 集計対象外
            [step("ACTIVE", only_active, 0), {"status": "INACTIVE"}],
            [step("INACTIVE", [], 0, action=[0]), {"status": "INACTIVE"}],
        ],
    }
    path = tmp_path / "ep.json"
    path.write_text(json.dumps(ep))
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    mine_episode(str(path), counts, None)
    assert dict(counts["T"]) == {2: [1, 1]}
