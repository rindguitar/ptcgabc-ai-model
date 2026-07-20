"""ko_threat（§48・エネ充足度で重み付けした KO 脅威推定）のテスト.

メタ・盤面とも軽量フェイク（SimpleNamespace）で組み、判定ロジックを直接検証する。
カード ID・タイプ・数値はすべてダミー（規約: Pokémon Elements を持ち込まない）。
"""

import os
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

pytest.importorskip("cg.api", reason="cabt Engine (cg) が見つからない")

from agents import ko_threat  # noqa: E402


def _meta(damage, cost, card_attacks, weakness=None, pokemon_type=None):
    """ko_threat が参照するフィールドだけ持つ CardMeta フェイク."""
    dmg = dict(damage)
    return NS(
        attack_damage=lambda aid: dmg.get(aid, 0),
        attack_cost=dict(cost),
        card_attacks=dict(card_attacks),
        weakness=dict(weakness or {}),
        pokemon_type=dict(pokemon_type or {}),
    )


def _pk(cid, hp, energy_n):
    return NS(id=cid, hp=hp, energies=[0] * energy_n)


def _side(active=None, bench=()):
    return NS(active=[active] if active else [], bench=list(bench))


# ダミー構成: カード1 はワザ100（必要エネ2 = attackId 10）を持つ。受け側はカード9（HP100）。
_M = _meta(damage={10: 100}, cost={10: 2}, card_attacks={1: [10]})


def test_ready_attacker_is_full_threat():
    """エネ充足＋打点が残 HP に届く → 脅威 1.0."""
    att = _side(active=_pk(1, 120, energy_n=2))
    dfd = _side(active=_pk(9, 100, energy_n=0))
    assert ko_threat(att, dfd, _M) == 1.0


def test_one_energy_short_is_discounted():
    """エネ1不足（次の1付与で打てる＝育成中）→ 0.5・2不足以上 → 0."""
    dfd = _side(active=_pk(9, 100, energy_n=0))
    assert ko_threat(_side(active=_pk(1, 120, 1)), dfd, _M) == 0.5
    assert ko_threat(_side(active=_pk(1, 120, 0)), dfd, _M) == 0.0


def test_insufficient_damage_is_no_threat():
    """打点が残 HP に届かなければエネが足りていても脅威 0（残 HP 依存）."""
    att = _side(active=_pk(1, 120, energy_n=2))
    assert ko_threat(att, _side(active=_pk(9, 150, 0)), _M) == 0.0
    # ダメージを受けて残 HP が下がれば同じ盤面でも脅威化する
    assert ko_threat(att, _side(active=_pk(9, 80, 0)), _M) == 1.0


def test_bench_attacker_is_discounted():
    """ベンチの充足アタッカーは昇格が要るので 0.5（アクティブ充足なら 1.0 が勝つ）."""
    dfd = _side(active=_pk(9, 100, energy_n=0))
    att = _side(active=_pk(8, 50, 0), bench=[_pk(1, 120, 2)])  # アクティブは無ワザ
    assert ko_threat(att, dfd, _M) == 0.5


def test_weakness_doubles_effective_damage():
    """弱点一致でダメージ×2＝素の打点では届かない相手も KO 圏に入る."""
    m = _meta(
        damage={10: 60},
        cost={10: 2},
        card_attacks={1: [10]},
        weakness={9: 3},  # 受け側カード9 の弱点タイプ=3
        pokemon_type={1: 3},  # 攻め側カード1 のタイプ=3（一致）
    )
    att = _side(active=_pk(1, 120, energy_n=2))
    dfd = _side(active=_pk(9, 100, energy_n=0))
    assert ko_threat(att, dfd, m) == 1.0  # 60×2=120 ≥ 100
    m.pokemon_type[1] = 4  # タイプ不一致なら 60 < 100 で脅威なし
    assert ko_threat(att, dfd, m) == 0.0


def test_empty_sides_are_zero():
    """受け側アクティブ不在・攻め側不在は脅威 0（クラッシュしない）."""
    att = _side(active=_pk(1, 120, 2))
    assert ko_threat(att, _side(), _M) == 0.0
    assert ko_threat(_side(), _side(active=_pk(9, 100, 0)), _M) == 0.0
    assert ko_threat(None, None, _M) == 0.0


def test_wrap_threat_bonus_shifts_value():
    """wrap_threat_bonus: 相手だけが脅威なら v が α だけ下がる（対称差・clamp あり）."""
    pytest.importorskip("torch", reason="torch が無い環境（ホスト）では skip")
    from nn_eval import wrap_threat_bonus

    def base(_obs):
        return 0.5, [1.0]

    att = _side(active=_pk(1, 120, energy_n=2))  # 相手（充足）
    dfd = _side(active=_pk(9, 100, energy_n=0))  # 自分（無ワザ＝脅威 0）
    obs = NS(current=NS(yourIndex=0, players=[dfd, att]), select=None)
    v, _ = wrap_threat_bonus(base, _M, 0.3)(obs)
    assert v == pytest.approx(0.2)  # 0.5 − 0.3×(1.0−0.0)
