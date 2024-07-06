import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """

    def person_probablility(person, gene_count, has_trait):
        def inherits(parent):
            parent_count = 2 if parent in two_genes else (
                1 if parent in one_gene else 0)

            chance = 1
            if parent_count == 0:
                chance *= PROBS["mutation"]
            elif parent_count == 1:
                chance *= 0.5
            else:
                chance *= (1-PROBS["mutation"])

            return chance

        mother, father = people[person]["mother"], people[person]["father"]
        if not mother and not father:
            return PROBS["gene"][gene_count] * PROBS["trait"][gene_count][has_trait]

        mother_chance = inherits(mother)
        father_chance = inherits(father)

        if gene_count == 0:
            result = (1-mother_chance) * (1-father_chance)
        elif gene_count == 1:
            result = (mother_chance * (1-father_chance) +
                      (1-mother_chance) * father_chance)
        else:
            result = mother_chance * father_chance

        return result * PROBS["trait"][gene_count][has_trait]

    result = 1
    zero_genes = set()
    for person in people:
        if person not in one_gene and person not in two_genes:
            zero_genes.add(person)

    for person in zero_genes:
        result *= person_probablility(person, 0, person in have_trait)

    for person in one_gene:
        result *= person_probablility(person, 1, person in have_trait)

    for person in two_genes:
        result *= person_probablility(person, 2, person in have_trait)

    return result


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    def update_person(person, gene_count, has_trait):
        probabilities[person]["gene"][gene_count] += p
        probabilities[person]["trait"][has_trait] += p

    zero_genes = set()
    for person in probabilities:
        if person not in one_gene and person not in two_genes:
            zero_genes.add(person)

    for person in zero_genes:
        update_person(person, 0, person in have_trait)

    for person in one_gene:
        update_person(person, 1, person in have_trait)

    for person in two_genes:
        update_person(person, 2, person in have_trait)


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        for parameter in probabilities[person]:
            current_sum = sum(probabilities[person][parameter].values())
            for option in probabilities[person][parameter]:
                probabilities[person][parameter][option] /= current_sum


if __name__ == "__main__":
    # print(joint_probability({
    #     'Harry': {'name': 'Harry', 'mother': 'Lily', 'father': 'James', 'trait': None},
    #     'James': {'name': 'James', 'mother': None, 'father': None, 'trait': True},
    #     'Lily': {'name': 'Lily', 'mother': None, 'father': None, 'trait': False}
    # }, {"Harry"}, {"James"}, {"James"}))

    main()
