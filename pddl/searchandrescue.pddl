(define (domain searchandrescue)
    (:requirements :typing)
    (:types robot person wall hospital location direction)
    (:constants
        up down left right - direction
    )
    (:predicates
        (conn ?v0 - location ?v1 - location ?v2 - direction)
        (clear ?v0 - location)
        (robot-at ?v0 - robot ?v1 - location)
        (person-at ?v0 - person ?v1 - location)
        (wall-at ?v0 - wall ?v1 - location)
        (hospital-at ?v0 - hospital ?v1 - location)
        (carrying ?v0 - robot ?v1 - person)
        (handsfree ?v0 - robot)
    )

    (:action move-robot
        :parameters (?robot - robot ?from - location ?to - location ?dir - direction)
        :precondition (and
            (conn ?from ?to ?dir)
            (robot-at ?robot ?from)
            (clear ?to)
        )
        :effect (and
            (not (robot-at ?robot ?from))
            (robot-at ?robot ?to)
            (not (clear ?to))
            (clear ?from)
        )
    )

    (:action pickup-person
        :parameters (?robot - robot ?person - person ?loc - location)
        :precondition (and
            (robot-at ?robot ?loc)
            (person-at ?person ?loc)
            (handsfree ?robot)
        )
        :effect (and
            (not (person-at ?person ?loc))
            (not (handsfree ?robot))
            (carrying ?robot ?person)
        )
    )

    (:action dropoff-person
        :parameters (?robot - robot ?person - person ?loc - location)
        :precondition (and
            (carrying ?robot ?person)
            (robot-at ?robot ?loc)
        )
        :effect (and
            (person-at ?person ?loc)
            (handsfree ?robot)
            (not (carrying ?robot ?person))
        )
    )

)
        